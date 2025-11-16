#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score-Taxonomy Mapper – v5

Diese Version erweitert den bestehenden v4e-Mapper um die Zuordnung von
``Maschinentyp``, ``Werkstoff`` und ``Anwendungsbereich`` für jedes
Produkt. Ziel ist es, möglichst viele Artikel ohne KI-Unterstützung
einwandfrei zu klassifizieren, indem vorhandene Katalogdaten,
Synonyme und regelbasierte Heuristiken genutzt werden. Ein LLM-Fallback
kann optional per Flag zugeschaltet werden und kommt nur zum Einsatz,
wenn alle heuristischen Ansätze versagt haben.

Hauptpunkte:
  * Lädt eine optionale ``taxonomy.yaml`` mit Definitionen von
    Kategorien, Maschinentypen, Werkstoffen und Anwendungsbereichen.
  * Lädt ``synonyms_rules.yaml`` mit Aliassen und Stopwörtern für
    Kategorien/Werkstoffe/Anwendungen.
  * Liefert zusätzliche Spalten ``Maschine``, ``Werkstoff`` und
    ``Anwendung`` mit Quellenangabe.
  * Erstellt Reports über unbekannte Tokens, die im nächsten Lauf zur
    Verbesserung der Synonyme genutzt werden können (iteratives Lernen).
  * Optionaler LLM-Fallback via ``--llm-enabled`` Flag (Standard: False).

Die grundlegende Logik für Körnungserkennung und PDF-Indizierung wurde
aus v4e übernommen und nur minimal angepasst.

Hinweis:
  * Ohne ``--llm-enabled`` wird KEIN OpenAI-Aufruf getätigt.
  * Die Datei ist so geschrieben, dass sie in deinem bisherigen
    Projektordner ``C:\\Python\\Filter`` direkt lauffähig ist.
"""

import argparse
import datetime as dt
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPORT_DIR = HERE / "_export"
CACHE_PDF_INDEX = HERE / "_cache_pdf_index.json"
CHECKPOINT = HERE / "checkpoint_latest.csv"

# RapidFuzz (optional, aber empfohlen)
try:
    from rapidfuzz import fuzz, process

    HAVE_RF = True
except Exception:
    HAVE_RF = False


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def ts_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def println(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# YAML / Konfiguration
# ---------------------------------------------------------------------------


def read_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def newest_matching(pattern: str) -> Optional[Path]:
    cand = sorted(HERE.glob(pattern))
    return cand[-1] if cand else None


def try_load_yaml_candidates(names: List[str]) -> Tuple[dict, Optional[Path]]:
    for n in names:
        p = HERE / n
        if p.exists():
            try:
                return read_yaml(p), p
            except Exception as e:
                println(f"[WARN] YAML '{p.name}' konnte nicht geladen werden: {e}")
    return {}, None


def load_rules():
    # Normalisierung
    normalize_cfg, normalize_path = try_load_yaml_candidates(["normalize_rules.yaml"])

    # Synonyme
    syn_path = newest_matching("synonyms_rules*.yaml") or (HERE / "synonyms_rules.yaml")
    synonyms_cfg = read_yaml(syn_path) if syn_path and syn_path.exists() else {}

    # PDF-Regeln (optional, derzeit nur für Serien/Körnung)
    pdf_rules = read_yaml(HERE / "pdf_extract_rules.yaml") if (HERE / "pdf_extract_rules.yaml").exists() else {}

    # Taxonomie (Maschine/Werkstoff/Anwendung etc.)
    tax_path = HERE / "taxonomy.yaml"
    taxonomy = read_yaml(tax_path) if tax_path.exists() else {}

    return normalize_cfg, synonyms_cfg, pdf_rules, taxonomy, {
        "normalize_path": normalize_path,
        "synonyms_path": syn_path,
        "taxonomy_path": tax_path,
    }


# ---------------------------------------------------------------------------
# Normalisierung / Aliasse / Stopwörter
# ---------------------------------------------------------------------------


def apply_normalize_rules(s: str, cfg: dict) -> str:
    """Unterstützt: strip_chars, replace, regex_replace, collapse_spaces, lowercase."""
    if not s:
        return ""
    out = s

    # strip_chars: 1-Zeichen via translate, >1-Zeichen via regex-replace
    strip_chars = (cfg or {}).get("strip_chars")
    if isinstance(strip_chars, (list, tuple)):
        single = [c for c in strip_chars if isinstance(c, str) and len(c) == 1]
        multi = [c for c in strip_chars if isinstance(c, str) and len(c) > 1]
        if single:
            table = str.maketrans({c: " " for c in single})
            out = out.translate(table)
        for tok in multi:
            try:
                out = re.sub(re.escape(tok), " ", out)
            except re.error:
                out = out.replace(tok, " ")

    # replace (als Regex, case-insensitive)
    replace_map = (cfg or {}).get("replace")
    if isinstance(replace_map, dict):
        for pat, repl in replace_map.items():
            try:
                out = re.sub(str(pat), str(repl), out, flags=re.IGNORECASE)
            except re.error:
                out = out.replace(str(pat), str(repl))

    # regex_replace (in Reihenfolge)
    rx = (cfg or {}).get("regex_replace")
    if isinstance(rx, list):
        for rule in rx:
            if isinstance(rule, dict) and "pattern" in rule and "repl" in rule:
                pattern = rule["pattern"]
                repl = rule["repl"]
                try:
                    out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
                except re.error as e:
                    println(f"[WARN] regex_replace ignoriert (invalid): {pattern} -> {e}")

    # collapse_spaces
    if bool((cfg or {}).get("collapse_spaces", True)):
        out = re.sub(r"\s+", " ", out).strip()

    # lowercase
    if bool((cfg or {}).get("lowercase", False)):
        out = out.lower()

    return out


def apply_alias_global(s: str, syn_cfg: dict) -> str:
    """Globale Alias-Ersetzung (alias_to_canonical)."""
    if not s:
        return ""
    out = s
    alias = (syn_cfg or {}).get("alias_to_canonical") or {}
    if isinstance(alias, dict):
        for pat, repl in alias.items():
            try:
                out = re.sub(pat, repl, out, flags=re.IGNORECASE)
            except re.error:
                out = out.replace(pat, repl)
    return out


def apply_attribute_alias(attr_name: str, value: str, syn_cfg: dict) -> str:
    """Attribut-spezifische Aliasse (attribute_alias)."""
    if not value:
        return ""
    lookup = (syn_cfg or {}).get("attribute_alias", {}).get(attr_name)
    if not isinstance(lookup, dict):
        return value
    v_low = value.lower()
    for pat, repl in lookup.items():
        try:
            if re.fullmatch(pat, v_low, flags=re.IGNORECASE):
                return repl
        except re.error:
            if v_low == pat.lower():
                return repl
    return value


def get_stopwords(syn_cfg: dict) -> List[str]:
    sw = (syn_cfg or {}).get("stopwords", [])
    if not isinstance(sw, list):
        return []
    return [str(x).strip().lower() for x in sw if x]


# ---------------------------------------------------------------------------
# Körnung / Grit
# ---------------------------------------------------------------------------

FEPA_MIN = 8
FEPA_MAX = 2500


def is_valid_grit(pstr: str) -> bool:
    if not pstr or not isinstance(pstr, str):
        return False
    m = re.fullmatch(r"P(\d{1,4})", pstr.strip(), flags=re.I)
    if not m:
        return False
    v = int(m.group(1))
    return FEPA_MIN <= v <= FEPA_MAX


def grit_from_text(text: str) -> str:
    """Extrahiert Körnung aus freiem Text – ignoriert Maße und nackte Zahlen ohne Label."""
    if not text:
        return ""
    s = text

    # Maße ausschließen
    if re.search(r"\d+\s*[x×]\s*\d+", s, flags=re.I):
        s = re.sub(r"\d+\s*[x×]\s*\d+(?:\s*[x×]\s*\d+)?", " ", s, flags=re.I)

    # P### oder 'Korn 120', 'Grit 120' etc.
    m = re.search(r"\bP\s?([1-9]\d{1,3})\b(?!\s*(?:mm|cm))", s, flags=re.I)
    if m:
        p = f"P{m.group(1)}"
        return p if is_valid_grit(p) else ""

    m = re.search(r"\b(?:Korn(?:ung|größe)?|Grit)\s*[:=]?\s*([1-9]\d{1,3})\b(?!\s*(?:mm|cm))", s, flags=re.I)
    if m:
        p = f"P{m.group(1)}"
        return p if is_valid_grit(p) else ""

    return ""


def looks_like_page_number(context: str) -> bool:
    if not context:
        return False
    return bool(re.search(r"\b(?:Seite|Page|S\.)\s*\d{1,4}\b", context, flags=re.I))


# ---------------------------------------------------------------------------
# Kategorien (beschichtet vs. nicht beschichtet)
# ---------------------------------------------------------------------------


def load_coated_categories(taxonomy: dict) -> Tuple[set, set]:
    coated = set()
    non_coated = set()
    for k in (taxonomy.get("coated_categories") or []):
        coated.add(str(k).casefold())
    for k in (taxonomy.get("non_coated_categories") or []):
        non_coated.add(str(k).casefold())
    return coated, non_coated


# ---------------------------------------------------------------------------
# PDF-Index (Serien → Attribute)
# ---------------------------------------------------------------------------

SERIES_PATTERNS = [
    r"\b([A-Z]{1,3}\s?\d{2,4}[A-Z]{0,3})\b",
    r"\b(PS\s?\d{2,3}[A-Z]?)\b",
    r"\b(CS\s?\d{3,4}[A-Z]{0,2})\b",
    r"\b(LS\s?\d{3}[A-Z]?)\b",
    r"\b(SMT\s?\d{3})\b",
    r"\b(QMC\s?\d{3})\b",
    r"\b(QRC\s?\d{3})\b",
]


def extract_series_tokens(s: str) -> set:
    if not s:
        return set()
    tokens = set()
    for pat in SERIES_PATTERNS:
        for m in re.finditer(pat, s, flags=re.I):
            tokens.add(m.group(1).replace(" ", "").upper())
    return tokens


def build_pdf_index(pdf_dir: Path, force_rebuild: bool = False) -> dict:
    if CACHE_PDF_INDEX.exists() and not force_rebuild:
        try:
            with open(CACHE_PDF_INDEX, "r", encoding="utf-8") as f:
                idx = json.load(f)
                println(f"PDF-Index aus Cache: {len(idx)} Schlüssel.")
                return idx
        except Exception:
            pass

    println(f"PDF-Index wird aufgebaut aus {pdf_dir} … (Backend: pypdf)")
    index: Dict[str, dict] = {}
    from pypdf import PdfReader

    files: List[Path] = []
    for ext in ("*.pdf", "*.PDF", "*.xls*", "*.XLS*"):
        files.extend(list(pdf_dir.rglob(ext)))

    for fp in files:
        try:
            keyset = extract_series_tokens(fp.stem)
            text = ""
            if fp.suffix.lower() == ".pdf":
                try:
                    reader = PdfReader(str(fp))
                    pages = reader.pages[:4] if len(reader.pages) > 4 else reader.pages
                    for p in pages:
                        text += "\n" + (p.extract_text() or "")
                except Exception:
                    text = ""
            else:
                text = fp.stem

            low = text.lower()
            keyset |= extract_series_tokens(text)

            attrs: Dict[str, Any] = {}
            # Körnung aus dem Katalogtext (nur gelabelt)
            for lab in (r"k[öo]rn(?:ung|größe)", r"grit"):
                m = re.search(lab + r".{0,40}\bP?\s?([1-9]\d{1,3})\b", low, flags=re.I | re.S)
                if m:
                    p = f"P{m.group(1)}"
                    if is_valid_grit(p):
                        attrs["Körnung"] = p
                        attrs["Körnung_Source"] = "pdf"
                        break

            obj = {"file": str(fp), "text": low, "attrs": attrs}
            for k in keyset:
                index[k] = obj

        except Exception:
            continue

    with open(CACHE_PDF_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    println(f"PDF-Index: {len(index)} Schlüssel erfasst (Cache aktualisiert).")
    return index


# ---------------------------------------------------------------------------
# Spaltenerkennung
# ---------------------------------------------------------------------------

CAND_NAME = ["Artikelname", "Name", "title"]
CAND_DESC = ["Beschreibungen", "Beschreibung", "desc"]
CAND_CAT = ["Kat1", "Warengruppe", "category", "Kategorie"]
CAND_BRAND = ["Hersteller", "Brand", "Marke", "Hersteller Nr.", "Hersteller-Nr.", "brand"]
CAND_CODE = ["Artikelnummer", "SKU", "sku", "HAN", "han", "Hersteller Artikelnummer"]


def detect_cols(df: pd.DataFrame) -> dict:
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    cols = {
        "name": pick(CAND_NAME) or df.columns[0],
        "desc": pick(CAND_DESC),
        "cat": pick(CAND_CAT) or df.columns[min(2, len(df.columns) - 1)],
        "brand": pick(CAND_BRAND),
        "code": pick(CAND_CODE),
    }
    return cols


# ---------------------------------------------------------------------------
# Taxonomie-Matching (Maschine / Werkstoff / Anwendung)
# ---------------------------------------------------------------------------


def extract_tokens(text: str, stopwords: List[str]) -> List[str]:
    if not text:
        return []
    t = re.sub(r"[^a-z0-9äöüß ]+", " ", text.lower())
    parts = re.split(r"\s+", t)
    return [p for p in parts if p and p not in stopwords]


def best_match_from_list(
    text: str,
    candidates: List[str],
    stopwords: List[str],
    min_score: float = 75.0,
) -> Tuple[str, float]:
    """Fuzzy-Match eines Textes gegen eine Liste von Kandidaten."""
    if not candidates or not text:
        return "", 0.0
    if not HAVE_RF:
        # Sehr einfacher Fallback: Substring-Suche
        low = text.lower()
        for c in candidates:
            if c.lower() in low:
                return c, 80.0
        return "", 0.0

    result = process.extractOne(
        text,
        candidates,
        scorer=fuzz.token_set_ratio,
    )
    if not result:
        return "", 0.0
    label, score, _ = result
    if score < min_score:
        return "", float(score)
    return label, float(score)


def taxonomy_lists_from_cfg(taxonomy: dict) -> dict:
    """Zieht Listen für Maschine/Werkstoff/Anwendung aus taxonomy.yaml."""
    res = {
        "machine_main": [],
        "machine_sub": [],
        "material_main": [],
        "material_sub": [],
        "application_main": [],
        "application_sub": [],
    }

    machines = taxonomy.get("machines") or []
    for m in machines:
        main = str(m.get("name", "")).strip()
        if not main:
            continue
        res["machine_main"].append(main)
        for sub in m.get("subtypes") or []:
            sub_name = str(sub).strip()
            if sub_name:
                res["machine_sub"].append(sub_name)

    materials = taxonomy.get("werkstoffe") or []
    for w in materials:
        main = str(w.get("name", "")).strip()
        if not main:
            continue
        res["material_main"].append(main)
        for sub in w.get("subtypes") or []:
            sub_name = str(sub).strip()
            if sub_name:
                res["material_sub"].append(sub_name)

    applications = taxonomy.get("anwendungen") or []
    for a in applications:
        main = str(a.get("name", "")).strip()
        if not main:
            continue
        res["application_main"].append(main)
        for sub in a.get("subtypes") or []:
            sub_name = str(sub).strip()
            if sub_name:
                res["application_sub"].append(sub_name)

    return res


def apply_compat_rules(
    category: str,
    machine_sub: str,
    application_sub: str,
    taxonomy: dict,
) -> float:
    """
    Einfache Kompatibilitäts-Logik: Wenn eine Regel in ``compat_rules``
    verletzt wird, senken wir die Confidence etwas. Wenn sie passt,
    boosten wir leicht.
    """
    rules = taxonomy.get("compat_rules") or {}
    score_delta = 0.0
    cat_low = (category or "").casefold()
    m_low = (machine_sub or "").casefold()
    a_low = (application_sub or "").casefold()

    for rule in rules.get("category_machine_application", []):
        cat_match = [str(x).casefold() for x in rule.get("categories", [])]
        mach_match = [str(x).casefold() for x in rule.get("machines", [])]
        app_match = [str(x).casefold() for x in rule.get("applications", [])]
        delta = float(rule.get("delta", 0.0))

        if (not cat_match or cat_low in cat_match) and (
            not mach_match or m_low in mach_match
        ) and (not app_match or a_low in app_match):
            score_delta += delta

    return score_delta


# ---------------------------------------------------------------------------
# LLM-Fallback (optional, standardmäßig aus)
# ---------------------------------------------------------------------------


def llm_fallback_classification(
    base_text: str,
    taxonomy: dict,
    api_key: Optional[str],
) -> dict:
    """
    Optionaler LLM-Fallback, wenn jede Heuristik versagt.
    Standardmäßig deaktiviert (wird nur verwendet, wenn --llm-enabled
    und ein OPENAI_API_KEY gesetzt ist).
    """
    if not api_key:
        return {}

    # Import erst hier, damit es keine harte Abhängigkeit gibt,
    # falls du OpenAI nicht nutzen möchtest.
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        println(f"[WARN] OpenAI-SDK nicht verfügbar: {e}")
        return {}

    client = OpenAI(api_key=api_key)

    machines = taxonomy.get("machines") or []
    materials = taxonomy.get("werkstoffe") or []
    applications = taxonomy.get("anwendungen") or []

    def names(lst):
        return [m.get("name", "") for m in lst if m.get("name")]

    prompt = (
        "Du bist ein Klassifikations-Experte für Schleifmittel. "
        "Ordne folgenden Produkttext jeweils EINEM Maschinentyp, EINEM Werkstoff "
        "und EINEM Anwendungsbereich aus den gegebenen Listen zu. "
        "Antworte als kompaktes JSON mit den Keys "
        "Maschine_main, Maschine_sub, Werkstoff_main, Werkstoff_sub, "
        "Anwendung_main, Anwendung_sub.\n\n"
        f"TEXT:\n{base_text}\n\n"
        f"Maschinen (Hauptebene): {names(machines)}\n"
        f"Werkstoffe (Hauptebene): {names(materials)}\n"
        f"Anwendungen (Hauptebene): {names(applications)}\n"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        if not isinstance(data, dict):
            return {}
        return {
            "Maschine_main": data.get("Maschine_main", ""),
            "Maschine_sub": data.get("Maschine_sub", ""),
            "Werkstoff_main": data.get("Werkstoff_main", ""),
            "Werkstoff_sub": data.get("Werkstoff_sub", ""),
            "Anwendung_main": data.get("Anwendung_main", ""),
            "Anwendung_sub": data.get("Anwendung_sub", ""),
        }
    except Exception as e:
        println(f"[WARN] LLM-Fallback fehlgeschlagen: {e}")
        return {}


# ---------------------------------------------------------------------------
# Zeilenmapping
# ---------------------------------------------------------------------------


def best_series_key(row_text: str, code: str) -> Optional[str]:
    tok = set()
    tok |= extract_series_tokens(row_text)
    tok |= extract_series_tokens(code or "")
    return sorted(tok, key=lambda x: (-len(x), x))[0] if tok else None


def map_row(
    row: pd.Series,
    cols: dict,
    norm_cfg: dict,
    syn_cfg: dict,
    pdf_idx: dict,
    taxonomy_lists: dict,
    taxonomy: dict,
    stopwords: List[str],
    llm_enabled: bool,
    llm_api_key: Optional[str],
    unknown_counters: dict,
    coated_kat: set,
    non_coated_kat: set,
) -> dict:
    out: Dict[str, Any] = {}

    name = str(row.get(cols["name"], "") or "")
    desc = str(row.get(cols["desc"], "") or "")
    cat1 = str(row.get(cols["cat"], "") or "")
    brand = str(row.get(cols["brand"], "") or "")
    code = str(row.get(cols["code"], "") or "")
    sku = str(row.get("sku", row.get("Artikelnummer", row.get("SKU", ""))) or "")

    # Basistext für Matching
    base_raw = " | ".join([name, desc, brand, code, cat1])
    base_norm = apply_alias_global(apply_normalize_rules(base_raw, norm_cfg), syn_cfg)

    # Grund-Attribut-Container
    attrs = {
        "Kornart": "",
        "Kornart_Source": "",
        "Körnung": "",
        "Körnung_Source": "",
        "Unterlage": "",
        "Unterlage_Source": "",
        "Streuart": "",
        "Streuart_Source": "",
        "Bindung": "",
        "Bindung_Source": "",
        "Maschine_main": "",
        "Maschine_main_Source": "",
        "Maschine_sub": "",
        "Maschine_sub_Source": "",
        "Werkstoff_main": "",
        "Werkstoff_main_Source": "",
        "Werkstoff_sub": "",
        "Werkstoff_sub_Source": "",
        "Anwendung_main": "",
        "Anwendung_main_Source": "",
        "Anwendung_sub": "",
        "Anwendung_sub_Source": "",
    }

    # ----------------- 1) Körnung (wie v4e) -----------------
    grit_t = grit_from_text(name + " " + desc)
    if is_valid_grit(grit_t):
        attrs["Körnung"] = grit_t
        attrs["Körnung_Source"] = "title"

    pdf_key = best_series_key(base_norm, code) or ""
    if pdf_key and pdf_key in pdf_idx:
        out_pdf = pdf_idx[pdf_key]["attrs"]
        if not attrs["Körnung"] and is_valid_grit(out_pdf.get("Körnung", "")):
            attrs["Körnung"] = out_pdf["Körnung"]
            attrs["Körnung_Source"] = "pdf"

    # Konflikt: Titel gewinnt
    if attrs["Körnung"] and attrs["Körnung_Source"] != "title":
        grit_again = grit_from_text(name + " " + desc)
        if grit_again and grit_again != attrs["Körnung"]:
            text = pdf_idx.get(pdf_key, {}).get("text", "")
            if not looks_like_page_number(text):
                attrs["Körnung"] = grit_again
                attrs["Körnung_Source"] = "title"

    # Unterlage/Streuart nur bei beschichteten Kategorien
    if cat1.casefold() in coated_kat:
        low = (name + " " + desc).lower()
        if "unterlage" in low or "backing" in low:
            m = re.search(r"(unterlage|backing)\s*[:=\-]?\s*([a-z0-9\-\/ ]{2,30})", low, flags=re.I)
            if m:
                val = m.group(2).strip()
                val = apply_attribute_alias("Unterlage", val, syn_cfg)
                attrs["Unterlage"] = val
                attrs["Unterlage_Source"] = "title"
        if "streu" in low or "coating" in low:
            m = re.search(r"(streu(?:art|ung)|coating)\s*[:=\-]?\s*([a-zäöü \-]{2,20})", low, flags=re.I)
            if m:
                val = m.group(2).strip()
                val = apply_attribute_alias("Streuart", val, syn_cfg)
                attrs["Streuart"] = val
                attrs["Streuart_Source"] = "title"
    else:
        attrs["Unterlage"] = ""
        attrs["Streuart"] = ""
        attrs["Unterlage_Source"] = ""
        attrs["Streuart_Source"] = ""

    # Körnung final validieren
    if attrs["Körnung"] and not is_valid_grit(attrs["Körnung"]):
        attrs["Körnung"] = ""
        attrs["Körnung_Source"] = ""

    # ----------------- 2) Maschine/Werkstoff/Anwendung -----------------
    token_text = " ".join([name, desc, cat1, brand]).lower()

    # Maschine
    m_main, m_main_score = best_match_from_list(
        token_text, taxonomy_lists["machine_main"], stopwords, min_score=75.0
    )
    m_sub, m_sub_score = best_match_from_list(
        token_text, taxonomy_lists["machine_sub"], stopwords, min_score=70.0
    )

    if m_main:
        attrs["Maschine_main"] = m_main
        attrs["Maschine_main_Source"] = f"heuristic:{int(m_main_score)}"
    else:
        unknown_counters["machine_main"][cat1].update([token_text])

    if m_sub:
        attrs["Maschine_sub"] = m_sub
        attrs["Maschine_sub_Source"] = f"heuristic:{int(m_sub_score)}"
    else:
        unknown_counters["machine_sub"][cat1].update([token_text])

    # Werkstoff
    w_main, w_main_score = best_match_from_list(
        token_text, taxonomy_lists["material_main"], stopwords, min_score=75.0
    )
    w_sub, w_sub_score = best_match_from_list(
        token_text, taxonomy_lists["material_sub"], stopwords, min_score=70.0
    )

    if w_main:
        attrs["Werkstoff_main"] = w_main
        attrs["Werkstoff_main_Source"] = f"heuristic:{int(w_main_score)}"
    else:
        unknown_counters["material_main"][cat1].update([token_text])

    if w_sub:
        attrs["Werkstoff_sub"] = w_sub
        attrs["Werkstoff_sub_Source"] = f"heuristic:{int(w_sub_score)}"
    else:
        unknown_counters["material_sub"][cat1].update([token_text])

    # Anwendung
    a_main, a_main_score = best_match_from_list(
        token_text, taxonomy_lists["application_main"], stopwords, min_score=75.0
    )
    a_sub, a_sub_score = best_match_from_list(
        token_text, taxonomy_lists["application_sub"], stopwords, min_score=70.0
    )

    if a_main:
        attrs["Anwendung_main"] = a_main
        attrs["Anwendung_main_Source"] = f"heuristic:{int(a_main_score)}"
    else:
        unknown_counters["application_main"][cat1].update([token_text])

    if a_sub:
        attrs["Anwendung_sub"] = a_sub
        attrs["Anwendung_sub_Source"] = f"heuristic:{int(a_sub_score)}"
    else:
        unknown_counters["application_sub"][cat1].update([token_text])

    # Kompatibilitäts-Score (kleiner Boost / Malus)
    compat_delta = apply_compat_rules(cat1, attrs["Maschine_sub"], attrs["Anwendung_sub"], taxonomy)
    if compat_delta != 0.0:
        for key in ("Maschine_sub_Source", "Anwendung_sub_Source"):
            src = attrs.get(key) or ""
            if src.startswith("heuristic:"):
                try:
                    base_score = int(src.split(":", 1)[1])
                except Exception:
                    base_score = 0
                new_score = max(0, min(100, base_score + int(compat_delta)))
                attrs[key] = f"heuristic:{new_score}"

    # Optionaler LLM-Fallback:
    # nur wenn aktiviert und wenn Haupt-Ebenen leer sind.
    if llm_enabled and (not attrs["Maschine_main"] or not attrs["Werkstoff_main"] or not attrs["Anwendung_main"]):
        llm_res = llm_fallback_classification(base_raw, taxonomy, llm_api_key)
        for k, v in llm_res.items():
            if v and not attrs.get(k):
                attrs[k] = v
                attrs[k + "_Source"] = "llm"

    out.update(
        {
            "sku": sku,
            "brand": brand,
            "title": name,
            "desc": desc,
            "category": cat1,
            "code": code,
            "pdf_key": pdf_key,
        }
    )
    out.update(attrs)
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(args):
    EXPORT_DIR.mkdir(exist_ok=True)

    normalize_cfg, syn_cfg, pdf_rules, taxonomy, paths = load_rules()
    stopwords = get_stopwords(syn_cfg)
    taxonomy_lists = taxonomy_lists_from_cfg(taxonomy)
    coated_kat, non_coated_kat = load_coated_categories(taxonomy)

    pdf_idx = build_pdf_index(Path(args.pdf_dir), force_rebuild=not CACHE_PDF_INDEX.exists())

    println(f"Lade Master-Excel: {args.input} (Sheet {args.sheet})")
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    if "sku" not in df.columns:
        df["sku"] = df.get("Artikelnummer", df.index.astype(str))

    cols = detect_cols(df)

    n = len(df)
    chunk = int(args.chunk_size)
    results: List[dict] = []

    grit_conflicts = 0
    invalid_grit_rows = 0
    cleared_noncoated_attrs = 0

    # Counters für "iteratives Lernen": unbekannte Tokens
    unknown_counters = {
        "machine_main": defaultdict(Counter),
        "machine_sub": defaultdict(Counter),
        "material_main": defaultdict(Counter),
        "material_sub": defaultdict(Counter),
        "application_main": defaultdict(Counter),
        "application_sub": defaultdict(Counter),
    }

    llm_enabled = bool(args.llm_enabled)
    llm_api_key = os.environ.get("OPENAI_API_KEY") if llm_enabled else None
    if llm_enabled and not llm_api_key:
        println("[WARN] --llm-enabled ist gesetzt, aber OPENAI_API_KEY fehlt. LLM wird ignoriert.")
        llm_enabled = False

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        println(f"Chunk {start}-{end} …")
        part = df.iloc[start:end].copy()
        for _, row in part.iterrows():
            res = map_row(
                row,
                cols,
                normalize_cfg,
                syn_cfg,
                pdf_idx,
                taxonomy_lists,
                taxonomy,
                stopwords,
                llm_enabled,
                llm_api_key,
                unknown_counters,
                coated_kat,
                non_coated_kat,
            )

            if res["pdf_key"] and res["Körnung_Source"] == "title":
                pdf_attr = pdf_idx.get(res["pdf_key"], {}).get("attrs", {})
                if is_valid_grit(pdf_attr.get("Körnung", "")) and pdf_attr.get("Körnung") != res["Körnung"]:
                    grit_conflicts += 1

            if res["Körnung"] and not is_valid_grit(res["Körnung"]):
                invalid_grit_rows += 1

            if res["category"].casefold() in non_coated_kat and (res.get("Unterlage") or res.get("Streuart")):
                cleared_noncoated_attrs += 1
                res["Unterlage"] = ""
                res["Streuart"] = ""
                res["Unterlage_Source"] = ""
                res["Streuart_Source"] = ""

            results.append(res)

        # Checkpoint nach jedem Chunk
        pd.DataFrame(results).to_csv(CHECKPOINT, index=False, encoding="utf-8-sig")

    enriched = pd.DataFrame(results)

    # Coverage / QA
    coverage_rows = [
        ("rows_total", n),
        ("rows_with_pdfkey", int((enriched["pdf_key"] != "").sum())),
        ("koernung_filled", int((enriched["Körnung"] != "").sum())),
        ("koernung_from_title", int((enriched["Körnung_Source"] == "title").sum())),
        ("koernung_from_pdf", int((enriched["Körnung_Source"] == "pdf").sum())),
        ("invalid_grit_rows", invalid_grit_rows),
        ("grit_conflicts", grit_conflicts),
        ("cleared_noncoated_attrs", cleared_noncoated_attrs),
        ("unterlage_filled", int((enriched["Unterlage"] != "").sum())),
        ("streuart_filled", int((enriched["Streuart"] != "").sum())),
        ("maschine_main_filled", int((enriched["Maschine_main"] != "").sum())),
        ("maschine_sub_filled", int((enriched["Maschine_sub"] != "").sum())),
        ("werkstoff_main_filled", int((enriched["Werkstoff_main"] != "").sum())),
        ("werkstoff_sub_filled", int((enriched["Werkstoff_sub"] != "").sum())),
        ("anwendung_main_filled", int((enriched["Anwendung_main"] != "").sum())),
        ("anwendung_sub_filled", int((enriched["Anwendung_sub"] != "").sum())),
    ]
    coverage = pd.DataFrame(coverage_rows, columns=["metric", "value"])

    # Title≠Output
    def title_grit(s):
        return grit_from_text(s or "")

    tmp = enriched.copy()
    tmp["title_grit"] = (tmp["title"] + " " + tmp["desc"]).map(title_grit)
    mismatch = tmp[(tmp["title_grit"] != "") & (tmp["title_grit"] != tmp["Körnung"])]
    mismatch = mismatch[
        [
            "sku",
            "brand",
            "title",
            "category",
            "title_grit",
            "Körnung",
            "Körnung_Source",
            "pdf_key",
        ]
    ]

    bad = enriched[(enriched["Körnung"] != "") & (~enriched["Körnung"].str.fullmatch(r"P\d{1,4}", na=False))]
    bad = bad[["sku", "brand", "title", "category", "Körnung", "Körnung_Source", "pdf_key"]]

    # Unknown-Token-Reports (für iteratives Lernen)
    unknown_rows = []
    for key, cat_map in unknown_counters.items():
        for cat, counter in cat_map.items():
            for token, cnt in counter.most_common(50):
                unknown_rows.append(
                    {
                        "type": key,
                        "category": cat,
                        "token": token[:200],
                        "count": cnt,
                    }
                )
    unknown_df = pd.DataFrame(unknown_rows)

    stamp = ts_stamp()
    prefix = args.export_prefix or ""
    p_enr = EXPORT_DIR / f"{prefix}score_mapping_enriched_{stamp}.xlsx"
    p_rep = EXPORT_DIR / f"{prefix}score_mapping_report_{stamp}.xlsx"
    p_cov = EXPORT_DIR / f"{prefix}attributes_coverage_{stamp}.xlsx"

    with pd.ExcelWriter(p_enr, engine="openpyxl") as xw:
        enriched.to_excel(xw, sheet_name="enriched", index=False)

    with pd.ExcelWriter(p_rep, engine="openpyxl") as xw:
        coverage.to_excel(xw, sheet_name="Summary", index=False)
        mismatch.head(5000).to_excel(xw, sheet_name="Title≠Output", index=False)
        bad.head(5000).to_excel(xw, sheet_name="Ungueltige_Koernung", index=False)
        unknown_df.to_excel(xw, sheet_name="unknown_tokens", index=False)

    with pd.ExcelWriter(p_cov, engine="openpyxl") as xw:
        coverage.to_excel(xw, index=False)

    println("Export:")
    println(f"  {p_enr}")
    println(f"  {p_rep}")
    println(f"  {p_cov}")
    println(f"Fertig. Verarbeitet: {n} Zeilen. Checkpoint: {CHECKPOINT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sheet", type=int, default=0)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--chunk-size", type=int, default=8000)
    ap.add_argument("--export-prefix", default="v5_")
    ap.add_argument(
        "--llm-enabled",
        action="store_true",
        help="Optional: LLM-Fallback aktivieren (nur wenn OPENAI_API_KEY gesetzt ist).",
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
