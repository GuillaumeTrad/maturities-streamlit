import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import pdfplumber
import pandas as pd

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)



# -----------------------------
# Helpers: dates / numbers
# -----------------------------
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
NOOP_RE = re.compile(r"\b\d{2}/\d{6,7}-\d{2}\b")

def parse_date_ddmmyyyy(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    # accepte dd/mm/yyyy ou dd.mm.yyyy
    s = s.replace(".", "/")
    try:
        return datetime.strptime(s, "%d/%m/%Y")
    except Exception:
        return None

def format_date_dot(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%d/%m/%Y").strftime("%d.%m.%Y")
    except Exception:
        return s

def swiss_amount_to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = s.replace("’", "'")
    s = s.replace("'", "")
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def compute_duration(date_val: datetime, maturity: datetime) -> Tuple[int, float, str]:
    months = (maturity.year - date_val.year) * 12 + (maturity.month - date_val.month)
    if maturity.day < date_val.day:
        months -= 1
    months = max(months, 0)

    years = round(months / 12.0, 2)

    # texte lisible
    if months <= 12:
        duree_txt = f"{months} months"
    else:
        duree_txt = f"{years:.2f} years"

    return months, years, duree_txt



# -----------------------------
# Data model
# -----------------------------
@dataclass
class TradeRecord:
    date_valeur: Optional[str]
    maturity: Optional[str]
    deal_type: Optional[str]
    devise: Optional[str]
    montant: Optional[float]
    taux: Optional[float]
    borrower: Optional[str]
    lender: Optional[str]
    months: Optional[int]    
    years: Optional[float]     
    duree: Optional[str]
    house: Optional[str]
    client_final: Optional[str]
    house_side: Optional[str]
    parse_status: str
    missing_fields: str
    page_number: int


# -----------------------------
# Core parsing functions
# -----------------------------
def extract_maturity(page_text: str) -> Optional[str]:
    m = re.search(r"Echéance du\s+(\d{2}/\d{2}/\d{4})", page_text)
    return m.group(1) if m else None

def find_trade_starts(page_text: str) -> List[int]:
    return [m.start() for m in NOOP_RE.finditer(page_text)]

def split_into_segments(page_text: str, starts: List[int]) -> List[str]: # Pour chaque numéro d'opération, on le met dans une liste.
    if not starts:
        return []
    starts_sorted = sorted(starts)
    segments = []
    for i, st in enumerate(starts_sorted):
        en = starts_sorted[i + 1] if i + 1 < len(starts_sorted) else len(page_text)
        segments.append(page_text[st:en])
    return segments

def extract_trade_line(segment_text: str) -> Optional[str]:
    for line in segment_text.splitlines():
        if NOOP_RE.search(line):
            return line
    return None

def extract_trade_line_fields(segment_text: str) -> Dict[str, Any]:
    line = extract_trade_line(segment_text)
    if not line:
        return {"ok": False}

    norm = re.sub(r"[ \t]+", " ", line).strip()

    m = re.search(
        r"(?P<noop>\d{2}/\d{6,7}-\d{2})\s+.*?\s+"
        r"(?P<montant>[\d'’]+)\s+"
        r"(?P<devise>[A-Z]{3})\s+"
        r"(?P<taux>[\d.,]+)\s+.*?\s+"
        r"(?P<date_valid>\d{2}/\d{2}/\d{4})\b",
        norm
    )
    if not m:
        return {"ok": False, "trade_line": norm}

    return {
        "ok": True,
        "montant_raw": m.group("montant"),
        "devise": m.group("devise"),
        "taux_raw": m.group("taux"),
        "date_valid": m.group("date_valid"),
        "trade_line": norm,
    }


# -----------------------------
# Borrower/Lender parsing (robust block-based)
# -----------------------------
CODE_RE = re.compile(r"^\*{0,3}[A-Z0-9\*]{1,10}-[A-Z0-9\*]{1,10}$", re.IGNORECASE)

def _normalize_line(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()

def _extract_entities_from_block(lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Extract (codes, names) from a borrower/lender block.
    Robust to cases where PDF extraction flattens multiple entities onto one line, e.g.:
      "Borrower : *FZ1-ZH FINARBIT AG / ZURICH RENE-*L VILLE DE RENENS"
    """
    codes: List[str] = []
    names: List[str] = []

    for raw in lines:
        s = _normalize_line(raw)
        if not s:
            continue

        # remove leading label remnants
        s = re.sub(r"^(Borrower|Lender)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
        if not s:
            continue

        # normalize spaced dashes in codes like "CPC -BE" -> "CPC-BE"
        s = re.sub(r"\s*-\s*", "-", s)

        tokens = s.split()
        if not tokens:
            continue

        curr_code = None
        curr_name_tokens: List[str] = []

        def flush():
            nonlocal curr_code, curr_name_tokens
            if curr_code and curr_name_tokens:
                name = " ".join(curr_name_tokens).strip()
                if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", name):
                    codes.append(curr_code)
                    names.append(name)
            curr_code = None
            curr_name_tokens = []

        for tok in tokens:
            tok_clean = tok.replace(" ", "")

            # new entity starts when a token looks like a code
            if CODE_RE.match(tok_clean):
                flush()
                curr_code = tok_clean
            else:
                if curr_code is None:
                    continue
                curr_name_tokens.append(tok)

        flush()

    return codes, names


def _slice_between(lines: List[str], start_pat: str, end_pats: List[str]) -> List[str]:
    """
    Return lines after the line containing start_pat (inclusive),
    stopping before a line containing any of end_pats.
    """
    start_idx = None
    for i, line in enumerate(lines):
        if re.search(start_pat, line, flags=re.IGNORECASE):
            start_idx = i
            break
    if start_idx is None:
        return []

    out: List[str] = []
    for j in range(start_idx, len(lines)):
        line = lines[j]
        if j != start_idx and any(re.search(p, line, flags=re.IGNORECASE) for p in end_pats):
            break
        out.append(line)
    return out

def extract_counterparties(segment_text: str) -> Dict[str, Any]: #segment_text est la variable pour un trade
    """
    Robust approach:
    - borrower_block = lines from 'Borrower :' until 'Lender :'
    - lender_block   = lines from 'Lender :' until 'Payment instr.' OR end of segment
    Then extract entities in each block as CODE + NAME.
    borrower_name/lender_name:
      - default: last name in borrower block, first name in lender block
      (later refined by FINARBIT rule)
    """
    lines = segment_text.splitlines()

    borrower_block = _slice_between(
        lines,
        start_pat=r"\bBorrower\s*:",
        end_pats=[r"\bLender\s*:"]
    )
    lender_block = _slice_between(
        lines,
        start_pat=r"\bLender\s*:",
        end_pats=[r"\bPayment\s+instr\.", r"\bPayment\s+instr\b", r"\bZU\s+GUNSTEN\b", r"\bIBAN\b", r"\bREFERENZ\b", r"\bISIN\b", r"\bNo\s+opération\b", NOOP_RE.pattern]
    )

    borrower_codes, borrower_names = _extract_entities_from_block(borrower_block)
    lender_codes, lender_names = _extract_entities_from_block(lender_block)

    borrower_name = borrower_names[-1] if borrower_names else None
    lender_name = lender_names[0] if lender_names else None

    return {
        "borrower_name": borrower_name,
        "lender_name": lender_name,
        "borrower_entities": borrower_names,
        "lender_entities": lender_names,
        "borrower_codes": borrower_codes,
        "lender_codes": lender_codes,
    }


# -----------------------------
# Flags
# -----------------------------

def detect_house_info(borrower_entities: List[str], lender_entities: List[str]) -> Dict[str, Optional[str]]:
    """
    FINARBIT can appear in borrower or lender entities.
    Rule: FINARBIT is above its client => client_final is the entity right after FINARBIT in the same side list.
    """
    def find_after_finarbit(entities: List[str]) -> Optional[str]:
        for i, e in enumerate(entities):
            if "FINARBIT" in e.upper():
                if i + 1 < len(entities):
                    return entities[i + 1]
                return None
        return None

    borrower_has = any("FINARBIT" in e.upper() for e in borrower_entities)
    lender_has = any("FINARBIT" in e.upper() for e in lender_entities)

    if borrower_has and not lender_has:
        return {
            "house": "FINARBIT",
            "house_side": "Borrower",
            "client_final": find_after_finarbit(borrower_entities),
        }
    if lender_has and not borrower_has:
        return {
            "house": "FINARBIT",
            "house_side": "Lender",
            "client_final": find_after_finarbit(lender_entities),
        }
    if borrower_has and lender_has:
        # rare: pick the side where we actually have a client after FINARBIT
        bc = find_after_finarbit(borrower_entities)
        lc = find_after_finarbit(lender_entities)
        if bc and not lc:
            return {"house": "FINARBIT", "house_side": "Borrower", "client_final": bc}
        if lc and not bc:
            return {"house": "FINARBIT", "house_side": "Lender", "client_final": lc}
        return {"house": "FINARBIT", "house_side": "Unknown", "client_final": bc or lc}

    return {"house": None, "house_side": None, "client_final": None}


# -----------------------------
# Status
# -----------------------------
def compute_parse_status(fields: Dict[str, Any]) -> Tuple[str, str]:
    required = [
        ("date_valeur", fields.get("date_valeur")),
        ("maturity", fields.get("maturity")),
        ("devise", fields.get("devise")),
        ("montant", fields.get("montant")),
        ("taux", fields.get("taux")),
        ("borrower", fields.get("borrower")),
        ("lender", fields.get("lender")),
    ]
    missing = [name for name, val in required if val in (None, "", [])]
    if missing:
        return "CHECK", ", ".join(missing)
    return "OK", ""

def pick_by_code_skip(codes: List[str], names: List[str], skip_codes: List[str], prefer: str = "first") -> Optional[str]:
    skip_set = {c.upper() for c in skip_codes}
    pairs = [(c, n) for c, n in zip(codes, names) if c and n]

    # filtre les codes à ignorer
    filtered = [(c, n) for c, n in pairs if c.upper() not in skip_set]

    if not filtered:
        return None

    return filtered[0][1] if prefer == "first" else filtered[-1][1]


# -----------------------------
# Parse trade segment
# -----------------------------
def parse_trade_segment(segment_text: str, maturity: Optional[str], page_number: int) -> TradeRecord:
    line_fields = extract_trade_line_fields(segment_text)
    cp = extract_counterparties(segment_text)

    raw_date_valeur = line_fields.get("date_valid") if line_fields.get("ok") else None
    date_valeur = format_date_dot(raw_date_valeur)

    devise = line_fields.get("devise") if line_fields.get("ok") else None
    montant = swiss_amount_to_float(line_fields.get("montant_raw")) if line_fields.get("ok") else None
    taux = swiss_amount_to_float(line_fields.get("taux_raw")) if line_fields.get("ok") else None

    borrower_entities = cp.get("borrower_entities", [])
    lender_entities = cp.get("lender_entities", [])

    # Borrower/Lender displayed values (pre-house adjustment) and skip Tradition code
    borrower_codes = cp.get("borrower_codes", [])
    lender_codes = cp.get("lender_codes", [])
    borrower_entities = cp.get("borrower_entities", [])
    lender_entities = cp.get("lender_entities", [])

    # Skip Tradition house code if present
    SKIP_CODES = ["**LS-LS"]

    borrower = pick_by_code_skip(borrower_codes, borrower_entities, SKIP_CODES, prefer="last") or cp.get("borrower_name")
    lender = pick_by_code_skip(lender_codes, lender_entities, SKIP_CODES, prefer="first") or cp.get("lender_name")

    all_codes = [c.upper() for c in (borrower_codes + lender_codes) if c]
    deal_type = "PP/Obligataire" if "**LS-LS" in all_codes else "Loan"


    house_info = detect_house_info(borrower_entities, lender_entities)

    # If FINARBIT present, borrower/lender displayed should remain the real entities:
    # - if FINARBIT on borrower side, borrower should be client_final if available
    # - if FINARBIT on lender side, lender should be client_final if available
    if house_info.get("house") == "FINARBIT":
        cf = house_info.get("client_final")
        if house_info.get("house_side") == "Borrower" and cf:
            borrower = cf
        if house_info.get("house_side") == "Lender" and cf:
            lender = cf

    # duration
    months = None
    years = None
    duree = None

    if date_valeur and maturity:
        dv = parse_date_ddmmyyyy(date_valeur)
        mat = parse_date_ddmmyyyy(maturity)
        if dv and mat:
            months, years, duree = compute_duration(dv, mat)


    status, missing = compute_parse_status({
        "date_valeur": date_valeur,
        "maturity": maturity,
        "devise": devise,
        "montant": montant,
        "taux": taux,
        "borrower": borrower,
        "lender": lender,
    })

    return TradeRecord(
        date_valeur=date_valeur,
        maturity=format_date_dot(maturity),
        deal_type=deal_type,
        devise=devise,
        montant=montant,
        taux=taux,
        borrower=borrower,
        lender=lender,
        duree=duree,
        house=house_info.get("house"),
        client_final=house_info.get("client_final"),
        house_side=house_info.get("house_side"),
        parse_status=status,
        missing_fields=missing,
        page_number=page_number,
        months=months,
        years=years,
    )


# -----------------------------
# Page / PDF parsing
# -----------------------------
def parse_page(page_text: str, page_number: int) -> List[TradeRecord]:
    maturity = extract_maturity(page_text)
    starts = find_trade_starts(page_text)
    segments = split_into_segments(page_text, starts)
    return [parse_trade_segment(seg, maturity, page_number) for seg in segments]

def parse_pdf(path: str) -> pd.DataFrame:
    records: List[TradeRecord] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # layout=True helps keep lines/spacing more consistent, but our borrower/lender parsing
            # no longer depends on large spacing, so it's safe either way.
            text = page.extract_text(layout=True) or page.extract_text() or ""
            if not text.strip():
                continue
            records.extend(parse_page(text, page_number=i))

    df = pd.DataFrame([asdict(r) for r in records])

    cols_main = [
        "date_valeur", "maturity", "deal_type", "devise", "montant", "taux",
        "borrower", "lender", "months", "years", "duree"
    ]
    cols_adv = ["house", "client_final", "house_side", "parse_status", "missing_fields", "page_number"]
    cols = [c for c in cols_main + cols_adv if c in df.columns]
    if not df.empty:
        df = df[cols]
    return df


# -----------------------------
# CLI entry
# -----------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parsing_engine.py <path_to_pdf>")
        raise SystemExit(2)

    pdf_path = sys.argv[1]
    df = parse_pdf(pdf_path)
    print(df.head(50).to_string(index=False))

    out_xlsx = "echeances_extraites.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"\nSaved: {out_xlsx}")
