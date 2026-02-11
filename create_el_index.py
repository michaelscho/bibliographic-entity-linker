import sqlite3
import json
import os
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm

# Config
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "register"       
INDEX_DIR = BASE_DIR / "data" / "indices"
SOURCE_FILE = DATA_DIR / "vd_works.jsonl"
DB_NAME = os.getenv("VD18_DB_NAME", "vd18_fts.db")

def normalize_text(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize('NFKD', text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = text.replace("ſ", "s").replace("ß", "ss").replace("æ", "ae").replace("œ", "oe")
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_year(year_str: str) -> str:
    matches = re.findall(r'\b(1[4-9]\d{2})\b', str(year_str))
    return matches[0] if matches else ""

def _first_scalar(val) -> str:
    if isinstance(val, list):
        for v in val:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""
    if val is None:
        return ""
    return str(val).strip()

def extract_vd_id(vd_ids_list: list) -> str:
    if not vd_ids_list:
        return ""
    for vid in vd_ids_list:
        if "VD18" in str(vid):
            return str(vid)
    return str(vd_ids_list[0]) if vd_ids_list else ""

def extract_vd_id_from_record(data: dict) -> str:
    vd_ids = data.get("vd_ids")
    if isinstance(vd_ids, list) and vd_ids:
        return extract_vd_id(vd_ids)

    for key in ("vd18_number", "vd18", "vd17_number", "vd17", "vd16_number", "vd16"):
        val = _first_scalar(data.get(key))
        if val:
            return val
    return ""

def build_index():
    print(f"--- Building Index from {SOURCE_FILE} ---")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db_path = INDEX_DIR / DB_NAME
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("DROP TABLE IF EXISTS vd18")
    
    c.execute("""
        CREATE VIRTUAL TABLE vd18 USING fts5(
            search_text, 
            clean_author,
            clean_title,
            year UNINDEXED,
            place UNINDEXED,
            vd_id UNINDEXED,
            tokenize="trigram"
        )
    """)

    batch = []
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Indexing"):
            try:
                data = json.loads(line)
                
                title_raw = data.get("title", "") or data.get("normalized_title", "")
                author_raw = data.get("author_primary", "")
                if not author_raw:
                    authors = data.get("authors", [])
                    if isinstance(authors, list):
                        author_raw = " ".join(str(a) for a in authors if str(a).strip())
                    else:
                        author_raw = _first_scalar(authors)
                year_raw = data.get("year", "")

                places = data.get("publication_places", [])
                if not places:
                    places = data.get("place", [])
                if isinstance(places, str):
                    places = [places]
                place_raw = places[0] if places else ""

                clean_title = normalize_text(title_raw)
                clean_author = normalize_text(author_raw)
                clean_place = normalize_text(place_raw)
                year = extract_year(year_raw)
                vd_id = extract_vd_id_from_record(data)
                
                if not vd_id:
                    continue

                search_text = f"{clean_author} {clean_title} {year} {clean_place}"
                
                if len(search_text) < 3: continue

                batch.append((search_text, clean_author, clean_title, year, clean_place, vd_id))
                
                if len(batch) >= 10000:
                    c.executemany("INSERT INTO vd18 VALUES (?, ?, ?, ?, ?, ?)", batch)
                    conn.commit()
                    batch = []
            except Exception: continue

    if batch:
        c.executemany("INSERT INTO vd18 VALUES (?, ?, ?, ?, ?, ?)", batch)
        conn.commit()

    print("Optimizing...")
    c.execute("INSERT INTO vd18(vd18) VALUES('optimize')")
    conn.close()
    print(f"Index built at {db_path}")

if __name__ == "__main__":
    if SOURCE_FILE.exists():
        build_index()
    else:
        print(f"Missing: {SOURCE_FILE}")
