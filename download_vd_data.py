import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm


# CONFIGURATION
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "register"
OUTPUT_FILE = DATA_DIR / "vd_works.jsonl"

HF_DATASET_ID = "SBB/VD-Metadata"
HF_CONFIG_NAME = "VD-Bib-Metadata"


# Helper functions
def parse_list_field(val: Any) -> List[str]:
    """
    Turns "[Foo, Bar]" into ["Foo", "Bar"].
    If it's already a list, returns cleaned string elements.
    """
    if isinstance(val, list):
        return [str(v).strip() for v in val if isinstance(v, str) and v.strip()]

    if not isinstance(val, str):
        return []

    s = val.strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def normalize_title(title: Optional[str]) -> str:
    if not title:
        return ""
    return " ".join(str(title).lower().split())


def is_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    if isinstance(val, list) and len(val) == 0:
        return True
    return False


def scalarize(val: Any) -> Optional[str]:
    """
    Converts values to a JSON-friendly string (or None).
    """
    if val is None:
        return None
    if isinstance(val, float):
        if val != val:  # NaN check
            return None
        return str(val)
    if isinstance(val, (int, bool)):
        return str(val)
    if isinstance(val, list):
        # If HF returns lists for some fields, keep the first non-empty string
        for x in val:
            sx = scalarize(x)
            if sx:
                return sx
        return None
    s = str(val).strip()
    return s if s else None


def get_first_present(row: Dict[str, Any], candidates: List[str]) -> Any:
    """Return the first non-empty value from a list of candidate column names."""
    for k in candidates:
        if k in row and not is_empty(row.get(k)):
            return row.get(k)
    return None


def clean_pica_like_text(val: Any) -> Optional[str]:
    s = scalarize(val)
    if not s:
        return None
    return s.replace("$", " ").strip()


# Column candidates
CANDIDATES = {
    "ppn": [
        "0100 Pica-Produktionsnummer",
    ],
    "vd16": [
        "2190 VD16-Nummer",
    ],
    "vd17": [
        "2191 VD17-Nummer",
    ],
    "vd18": [
        "2192 VD18-Nummer",
    ],
    "title": [
        "245 Titel",
        "4000 Haupttitel, Titelzusatz, Verantwortlichkeitsangabe",
    ],
    "alt_titles": [
        "246 Alternativtitel",
    ],
    "year": [
        "264 Erscheinungsjahr",
        "1100 Erscheinungsdatum/Entstehungsdatum",
    ],
    "place": [
        "264 Erscheinungsort",
        "4040 Normierter Ort",
    ],
    "authors": [
        "3000 Person/Familie als 1. geistiger Schöpfer",
    ],
}


def map_row_to_first_schema(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    vd16 = scalarize(get_first_present(row, CANDIDATES["vd16"]))
    vd17 = scalarize(get_first_present(row, CANDIDATES["vd17"]))
    vd18 = scalarize(get_first_present(row, CANDIDATES["vd18"]))

    if not (vd16 or vd17 or vd18):
        return None

    ppn = scalarize(get_first_present(row, CANDIDATES["ppn"]))

    title_raw = get_first_present(row, CANDIDATES["title"])
    title = clean_pica_like_text(title_raw)
    norm_title = normalize_title(title)

    alt_raw = get_first_present(row, CANDIDATES["alt_titles"])
    alt_titles = parse_list_field(alt_raw)

    auth_raw = get_first_present(row, CANDIDATES["authors"])
    authors = parse_list_field(auth_raw)

    year_raw = get_first_present(row, CANDIDATES["year"])
    year = scalarize(year_raw)

    place_raw = get_first_present(row, CANDIDATES["place"])
    place = clean_pica_like_text(place_raw)

    return {
        "id": str(ppn) if ppn is not None else None,
        "vd16_number": vd16,
        "vd17_number": vd17,
        "vd18_number": vd18,
        "title": title,
        "normalized_title": norm_title,
        "alt_titles": alt_titles,
        "authors": authors,
        "year": year,
        "place": place,
        "source": "VD-Metadata",
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- Loading {HF_DATASET_ID} ({HF_CONFIG_NAME}) ---")
    ds = load_dataset(HF_DATASET_ID, HF_CONFIG_NAME, split="train", streaming=True)

    print(f"Writing to {OUTPUT_FILE}...")
    count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Processing"):
            try:
                rec = map_row_to_first_schema(row)
                if rec is None:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                continue

    print(f"Created {OUTPUT_FILE}")
    print(f"Total records: {count}")


if __name__ == "__main__":
    main()
