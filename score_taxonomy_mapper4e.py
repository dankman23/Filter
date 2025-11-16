# -*- coding: utf-8 -*-
"""
Score Taxonomy Mapper – v4e
Änderungen ggü. v4d:
- FIX: strip_chars unterstützt jetzt sicher auch Token > 1 Zeichen (ValueError-Fix).
- Robustere Normalisierung: replace/regex_replace bleiben, Reihenfolge gewahrt.
- Rest wie v4d: Körnungs-Priorität (Titel > Code > PDF mit Label), FEPA-Range, Maße/Seitennummern-Filter,
  Kategorie-Gating für Unterlage/Streuart, QA-Reports (Summary, Title≠Output, Ungueltige_Koernung).
"""

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process
    HAVE_RF = True
except Exception:
    HAVE_RF = False

HERE = Path(__file__).resolve().parent
EXPORT_DIR = HERE / "_export"
CACHE_PDF_INDEX = HERE / "_cache_pdf_index.json"
CHECKPOINT = HERE / "checkpoint_latest.csv"

def ts_stamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def println(msg: str):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---------------- YAML laden ----------------

def read_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def newest_matching(pattern: str) -> Path | None:
    cand = sorted(HERE.glob(pattern))
    return cand[-1] if cand else None

def try_load_yaml_candidates(names):
    for n in names:
        p = HERE / n
        if p.exists():
            try:
                return read_yaml(p), p
            except Exception as e:
                println(f"[WARN] YAML '{p.name}' konnte nicht geladen werden: {e}")
    return {}, None

def load_rules():
    normalize_cfg, normalize_path = try_load_yaml_candidates(["normalize_rules.yaml"])
    syn_path = newest_matching("synonyms_rules*.yaml") or (HERE / "synonyms_rules.yaml")
    synonyms_cfg = read_yaml(syn_path) if syn_path and syn_path.exists() else {}
    pdf_rules = read_yaml(HERE / "pdf_extract_rules.yaml") if (HERE / "pdf_extract_rules.yaml").exists() else {}
    taxonomy = read_yaml(HERE / "taxonomy.yaml") if (HERE / "taxonomy.yaml").exists() else {}
    return normalize_cfg, synonyms_cfg, pdf_rules, taxonomy, {
        "normalize_path": normalize_path,
        "synonyms_path": syn_path,
    }

# --------------- Normalisierung + Aliase ----------------

def apply_normalize_rules(s: str, cfg: dict) -> str:
    """Unterstützt: strip_chars (list), replace (dict), regex_replace (list[{pattern,repl}]), collapse_spaces, lowercase."""
    if not s:
        return ""
    out = s

    # strip_chars: 1-Zeichen via translate, >1-Zeichen via regex-replace
    strip_chars = (cfg or {}).get("strip_chars")
    if isinstance(strip_chars, (list, tuple)):
        single = [c for c in strip_chars if isinstance(c, str) and len(c) == 1]
        multi  = [c for c in strip_chars if isinstance(c, str) and len(c) > 1]
        if single:
            table = str.maketrans({c: " " for c in single})
            out = out.translate(table)
        for tok in multi:
            try:
                out = re.sub(re.escape(tok), " ", out)
            except re.error:
                # Fallback: simple replace
                out = out.replace(tok, " ")

    # replace (als Regex, case-insensitive)
    replace_map = (cfg or {}).get("replace")
    if isinstance(replace_map, dict):
        for pat, repl in replace_map.items():
            try:
                out = re.sub(str(pat), str(repl), out, flags=re.IGNORECASE)
            except re.error:
                # Fallback: literal
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

def apply_alias(s: str, syn_cfg: dict) -> str:
    if not s:
        return ""
    out = s
    alias = syn_cfg.get("alias") or {}
    if isinstance(alias, dict):
        for pat, repl in alias.items():
            try:
                out = re.sub(pat, repl, out, flags=re.IGNORECASE)
            except re.error:
                out = out.replace(pat, repl)
    return out

# --------------- Körnung/Validierung -------------------

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
    """Extrahiert Körnung aus freiem Text – ignoriert Maße (123x98x10), 'mm', und nackte Zahlen ohne Label."""
    if not text:
        return ""
    s = text
    # Maße ausschließen
    if re.search(r"\d+\s*[x×]\s*\d+", s, flags=re.I):
        s = re.sub(r"\d+\s*[x×]\s*\d+(?:\s*[x×]\s*\d+)?", " ", s, flags=re.I)

    # P### oder 'grit ###' (ohne mm/cm dahinter)
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

# --------------- Kategorien ----------------------------

COATED_KAT = {k.casefold() for k in [
    "Schleifbänder",
    "Schleifpapier / Schleifrollen / Schleifbögen",
    "Schleifschwämme / Schleifklötze",
    "Gitterscheiben",
    "Fiberscheiben",
    "Vliesbänder",
    "Vliesrollen",
    "Vlies-Bögen / Handpads",
    "Vliesscheiben / Fächerscheiben / Schleifmopteller",
]}

NON_COATED_KAT = {k.casefold() for k in [
    "Trennscheiben", "Diamanttrennscheiben",
    "Fräser / Frässtifte", "Schleifstifte", "Bohrer / Lochwerkzeuge",
    "Technische Bürsten", "Pinselbürsten mit Schaft", "Topfbürsten mit Schaft",
    "Rundbürsten mit Schaft", "Kegelbürsten mit Schaft", "Kegelbürsten mit Gewinde",
    "Rundbürsten", "Topfbürsten mit Gewinde", "Tellerbürsten Composite",
    "Schleifsteine", "Schruppscheiben", "Schleifbockscheiben",
    "Schleifsterne", "Fächerschleifer / Schleifmops", "Fächerräder",
    "Schnellwechselscheiben", "Spezielle Schleifscheiben", "Grobreinigungsscheiben",
    "Feilen", "Schlüsselfeilen", "Handbürsten", "Innenbürsten",
    "Stichsägeblätter", "Säbelsägeblätter", "Kreissägeblätter"
]}

# --------------- PDF-Index -----------------------------

SERIES_PATTERNS = [
    r"\b([A-Z]{1,3}\s?\d{2,4}[A-Z]{0,3})\b",
    r"\b(PS\s?\d{2,3}[A-Z]?)\b",
    r"\b(CS\s?\d{3,4}[A-Z]{0,2})\b",
    r"\b(LS\s?\d{3}[A-Z]?)\b",
    r"\b(SMT\s?\d{3})\b",
    r"\b(QMC\s?\d{3})\b",
    r"\b(QRC\s?\d{3})\b",
]

def extract_series_tokens(s: str) -> set[str]:
    if not s:
        return set()
    tokens = set()
    for pat in SERIES_PATTERNS:
        for m in re.finditer(pat, s, flags=re.I):
            tokens.add(m.group(1).replace(" ", "").upper())
    return tokens

def build_pdf_index(pdf_dir: Path, force_rebuild=False) -> dict:
    if CACHE_PDF_INDEX.exists() and not force_rebuild:
        try:
            with open(CACHE_PDF_INDEX, "r", encoding="utf-8") as f:
                idx = json.load(f)
                println(f"PDF-Index aus Cache: {len(idx)} Schlüssel.")
                return idx
        except Exception:
            pass

    println(f"PDF-Index wird aufgebaut aus {pdf_dir} … (Backend: pypdf)")
    index = {}
    from pypdf import PdfReader

    files = []
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

            # nur "gelabelte" Körnung extrahieren
            attrs = {}
            for lab in (r"k[öo]rn(?:ung|größe)", r"grit"):
                m = re.search(lab + r".{0,40}\bP?\s?([1-9]\d{1,3})\b", low, flags=re.I|re.S)
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

# --------------- Spaltenerkennung ----------------------

CAND_NAME = ["Artikelname", "Name", "title"]
CAND_DESC = ["Beschreibungen", "Beschreibung", "desc"]
CAND_CAT  = ["Kat1", "Warengruppe", "category", "Kategorie"]
CAND_BRAND= ["Hersteller", "Brand", "Marke", "Hersteller Nr.", "Hersteller-Nr.", "brand"]
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
        "cat":  pick(CAND_CAT)  or df.columns[min(2, len(df.columns)-1)],
        "brand":pick(CAND_BRAND),
        "code": pick(CAND_CODE),
    }
    return cols

# --------------- Mapping -------------------------------

def best_series_key(row_text: str, code: str) -> str | None:
    tok = set()
    tok |= extract_series_tokens(row_text)
    tok |= extract_series_tokens(code or "")
    return sorted(tok, key=lambda x: (-len(x), x))[0] if tok else None

def map_row(row: pd.Series, cols: dict, norm_cfg: dict, syn_cfg: dict, pdf_idx: dict):
    out = {}
    name = str(row.get(cols["name"], "") or "")
    desc = str(row.get(cols["desc"], "") or "")
    cat1 = str(row.get(cols["cat"], "") or "")
    brand= str(row.get(cols["brand"], "") or "")
    code = str(row.get(cols["code"], "") or "")
    sku  = str(row.get("sku", row.get("Artikelnummer", row.get("SKU", ""))) or "")

    base = " | ".join([name, desc, brand, code, cat1])
    base = apply_alias(apply_normalize_rules(base, norm_cfg), syn_cfg)

    attrs = {
        "Kornart": "", "Kornart_Source": "",
        "Körnung": "", "Körnung_Source": "",
        "Unterlage": "", "Unterlage_Source": "",
        "Streuart": "", "Streuart_Source": "",
        "Bindung": "", "Bindung_Source": "",
    }

    # 1) Körnung aus Titel/Beschreibung (Top-Priorität)
    grit_t = grit_from_text(name + " " + desc)
    if is_valid_grit(grit_t):
        attrs["Körnung"] = grit_t
        attrs["Körnung_Source"] = "title"

    # 2) Serien-/PDF-Hints
    pdf_key = best_series_key(base, code) or ""
    if pdf_key and pdf_key in pdf_idx:
        out_pdf = pdf_idx[pdf_key]["attrs"]
        if not attrs["Körnung"] and is_valid_grit(out_pdf.get("Körnung", "")):
            attrs["Körnung"] = out_pdf["Körnung"]
            attrs["Körnung_Source"] = "pdf"

    # 3) Konflikt: Titel gewinnt (und Seitenzahlen-Kontext aussortieren)
    if attrs["Körnung"] and attrs["Körnung_Source"] != "title":
        grit_again = grit_from_text(name + " " + desc)
        if grit_again and grit_again != attrs["Körnung"]:
            text = pdf_idx.get(pdf_key, {}).get("text", "")
            if not looks_like_page_number(text):
                attrs["Körnung"] = grit_again
                attrs["Körnung_Source"] = "title"

    # 4) Unterlage/Streuart nur bei beschichteten Kategorien
    if cat1.casefold() in COATED_KAT:
        low = (name + " " + desc).lower()
        if "unterlage" in low or "backing" in low:
            m = re.search(r"(unterlage|backing)\s*[:=\-]?\s*([a-z0-9\-\/ ]{2,30})", low, flags=re.I)
            if m:
                attrs["Unterlage"] = m.group(2).strip().upper()
                attrs["Unterlage_Source"] = "title"
        if "streu" in low or "coating" in low:
            m = re.search(r"(streu(?:art|ung)|coating)\s*[:=\-]?\s*([a-zäöü \-]{2,20})", low, flags=re.I)
            if m:
                val = m.group(2).strip().lower()
                wl = {"offen","halb-offen","halboffen","geschlossen","dicht","offen gestreut","halb offen","close coat","open coat"}
                if val in wl:
                    attrs["Streuart"] = val
                    attrs["Streuart_Source"] = "title"
    else:
        attrs["Unterlage"] = ""
        attrs["Streuart"] = ""
        attrs["Unterlage_Source"] = ""
        attrs["Streuart_Source"] = ""

    # 5) Körnung final validieren
    if attrs["Körnung"] and not is_valid_grit(attrs["Körnung"]):
        attrs["Körnung"] = ""
        attrs["Körnung_Source"] = ""

    out.update({
        "sku": sku, "brand": brand, "title": name, "desc": desc, "category": cat1,
        "code": code, "pdf_key": pdf_key
    })
    out.update(attrs)
    return out

# --------------- Pipeline --------------------------------

def run(args):
    EXPORT_DIR.mkdir(exist_ok=True)
    normalize_cfg, syn_cfg, pdf_rules, taxonomy, paths = load_rules()
    pdf_idx = build_pdf_index(Path(args.pdf_dir), force_rebuild=not CACHE_PDF_INDEX.exists())

    println(f"Lade Master-Excel: {args.input} (Sheet {args.sheet})")
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    if "sku" not in df.columns:
        df["sku"] = df.get("Artikelnummer", df.index.astype(str))

    cols = detect_cols(df)

    n = len(df)
    chunk = int(args.chunk_size)
    results = []
    grit_conflicts = 0
    invalid_grit_rows = 0
    cleared_noncoated_attrs = 0

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        println(f"Chunk {start}-{end} …")
        part = df.iloc[start:end].copy()
        for _, row in part.iterrows():
            res = map_row(row, cols, normalize_cfg, syn_cfg, pdf_idx)

            if res["pdf_key"] and res["Körnung_Source"] == "title":
                pdf_attr = pdf_idx.get(res["pdf_key"], {}).get("attrs", {})
                if is_valid_grit(pdf_attr.get("Körnung","")) and pdf_attr.get("Körnung") != res["Körnung"]:
                    grit_conflicts += 1

            if res["Körnung"] and not is_valid_grit(res["Körnung"]):
                invalid_grit_rows += 1

            if res["category"].casefold() in NON_COATED_KAT and (res.get("Unterlage") or res.get("Streuart")):
                cleared_noncoated_attrs += 1
                res["Unterlage"] = ""
                res["Streuart"] = ""
                res["Unterlage_Source"] = ""
                res["Streuart_Source"] = ""

            results.append(res)

        pd.DataFrame(results).to_csv(CHECKPOINT, index=False, encoding="utf-8-sig")

    enriched = pd.DataFrame(results)

    # Coverage / QA
    coverage = pd.DataFrame([
        ("rows_total", n),
        ("rows_with_pdfkey", int((enriched["pdf_key"]!="").sum())),
        ("koernung_filled", int((enriched["Körnung"]!="").sum())),
        ("koernung_from_title", int((enriched["Körnung_Source"]=="title").sum())),
        ("koernung_from_pdf", int((enriched["Körnung_Source"]=="pdf").sum())),
        ("invalid_grit_rows", invalid_grit_rows),
        ("grit_conflicts", grit_conflicts),
        ("cleared_noncoated_attrs", cleared_noncoated_attrs),
        ("unterlage_filled", int((enriched["Unterlage"]!="").sum())),
        ("streuart_filled", int((enriched["Streuart"]!="").sum())),
    ], columns=["metric","value"])

    # Title≠Output
    def title_grit(s):
        return grit_from_text(s or "")
    tmp = enriched.copy()
    tmp["title_grit"] = (tmp["title"] + " " + tmp["desc"]).map(title_grit)
    mismatch = tmp[(tmp["title_grit"]!="") & (tmp["title_grit"]!=tmp["Körnung"])]
    mismatch = mismatch[["sku","brand","title","category","title_grit","Körnung","Körnung_Source","pdf_key"]]

    bad = enriched[(enriched["Körnung"]!="") & (~enriched["Körnung"].str.fullmatch(r"P\d{1,4}", na=False))]
    bad = bad[["sku","brand","title","category","Körnung","Körnung_Source","pdf_key"]]

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
    ap.add_argument("--export-prefix", default="v4e_")
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
