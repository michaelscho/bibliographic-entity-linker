import sqlite3
import re
import json
import sys
import os
import unicodedata
import typing_extensions as typing
from pathlib import Path
from rapidfuzz import fuzz
import google.generativeai as genai

# Configuration
BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "data" / "indices" 
DB_NAME = os.getenv("VD18_DB_NAME", "vd18_fts.db")
TEST_SET_PATH = BASE_DIR / "test_set.json"

# LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
genai.configure(api_key=GOOGLE_API_KEY)

class MatchDecision(typing.TypedDict):
    best_match_vd_id: str
    confidence: str
    reasoning: str

model = genai.GenerativeModel("gemini-2.5-pro")

# Params
MIN_FTS_TOKEN_LEN = 3
STRICT_MAX_TOKENS = 4
BROAD_MAX_TOKENS = 8
MIN_ACCEPT_SCORE = 50 

# Scoring params
YEAR_BONUS_EXACT = 15    
YEAR_BONUS_CLOSE = 5     
AUTHOR_BONUS = 15
PLACE_BONUS = 10 
VD18_PREF_BONUS = 5  
ENTITY_BONUS = 5     

STOPWORDS = {
    "der", "die", "das", "und", "oder", "von", "zu", "im", "in", "an", "auf",
    "herrn", "herr", "georgii", "georg", "joh", "jac", "et", "cum", "de",
    "opus", "tomus", "pars", "liber", "tractatus", "dissertatio", "vol", "cap",
    "tit", "pag", "etc", "bey", "buchhändler", "allhier", "haben", "sind", "ist",
    "eine", "einer", "ein", "neue", "mr", "mr-", "aus", "dem", "den", "des",
    "stuck", "stück", "item", "theil", "band", "bände", "über", "gegen", "nach"
}

ABBREVIATIONS = {
    "evangel": "evangelische", "hist": "historie", "math": "mathematik", 
    "theol": "theologie", "jur": "juristische", "med": "medizin", 
    "phil": "philosophie", "bot": "botschaft",
}

# Helpers
def db_sanity_check(db_path: Path):
    print(f"\n🩺 DIAGNOSTIC CHECK")
    if not db_path.exists():
        print("   FATAL: Database file does not exist.")
        sys.exit(1)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c = conn.cursor()
    c.execute("SELECT count(*) FROM vd18")
    print(f"   Total Rows:    {c.fetchone()[0]}")
    conn.close()
    print("   DB Connection OK.\n")

def normalize_db_vd_id(vd_id: str) -> str:
    if not vd_id: return ""
    return re.sub(r"[^a-zA-Z0-9]", "", vd_id).lower()

def extract_vd18_id_from_gt(s: str) -> str:
    if not s: return ""
    m = re.search(r"VD188?\s*:?\s*([0-9A-Z]+)", s, re.IGNORECASE)
    if m: return f"vd18{m.group(1).lower()}"
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()

def _basic_normalize(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = text.replace("ſ", "s").replace("ß", "ss").replace("æ", "ae").replace("œ", "oe")
    text = re.sub(r"[-–—\.,:;]", " ", text) 
    text = re.sub(r"\b([a-z]{5,})(ns|s)\b", r"\1", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def expand_abbreviations(text: str) -> str:
    tokens = text.split()
    resolved = []
    for t in tokens:
        clean_t = t.replace(".", "")
        resolved.append(ABBREVIATIONS.get(clean_t, t))
    return " ".join(resolved)

def normalize_for_fts(text: str) -> str:
    text = _basic_normalize(text)
    text = expand_abbreviations(text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_for_fuzz(text: str) -> str:
    return _basic_normalize(text)

def build_query_string(bibl_item: dict):
    parts = []
    place = ""
    author = ""
    def get_val(key):
        val = bibl_item.get(key, "")
        if isinstance(val, list): return " ".join(str(v) for v in val)
        return str(val)
    if "author" in bibl_item: 
        author = get_val("author")
        parts.append(author)
    if "title" in bibl_item: parts.append(get_val("title"))
    if "year" in bibl_item: parts.append(get_val("year"))
    if "place" in bibl_item: place = get_val("place")
    return " ".join(parts), place, author

def extract_query_year(query: str) -> str | None:
    m = re.search(r"\b(1[4-9]\d{2})\b", query)
    return m.group(1) if m else None

def build_fts_queries(norm_query_fts: str) -> tuple[str, str]:
    toks = []
    for t in norm_query_fts.split():
        if t in STOPWORDS or t.isdigit() or len(t) < MIN_FTS_TOKEN_LEN: continue
        toks.append(t)
    toks = sorted(list(set(toks)), key=len, reverse=True)
    strict_q = " AND ".join(toks[:STRICT_MAX_TOKENS])
    broad_q = " OR ".join(toks[:BROAD_MAX_TOKENS])
    return strict_q, broad_q

# Run 1
def execute_run_1(conn, q_fts, strict_q, broad_q, query_year, norm_query_fuzz, query_place, query_author):
    def sql_search(fts_query, year):
        c = conn.cursor()
        if year and year.isdigit():
            y = int(year)
            sql = """SELECT clean_author, clean_title, year, place, vd_id FROM vd18 
                     WHERE vd18 MATCH ? ORDER BY CASE WHEN year = ? THEN 2 WHEN year = ? THEN 1 WHEN year = ? THEN 1 ELSE 0 END DESC, bm25(vd18) ASC LIMIT 500"""
            return c.execute(sql, (fts_query, str(y), str(y-1), str(y+1))).fetchall()
        else:
            sql = "SELECT clean_author, clean_title, year, place, vd_id FROM vd18 WHERE vd18 MATCH ? ORDER BY bm25(vd18) ASC LIMIT 500"
            return c.execute(sql, (fts_query,)).fetchall()

    def process(rows, strict_mode):
        out = []
        for db_author, db_title, db_year, db_place, vd_id in rows:
            if not vd_id:
                continue
            full_label = f"{db_author or ''} {db_title or ''}".strip()
            norm_db_label = normalize_for_fuzz(full_label)
            score = fuzz.token_set_ratio(norm_query_fuzz, norm_db_label)
            
            if query_year and db_year and str(db_year).isdigit():
                dy = abs(int(query_year) - int(db_year))
                if strict_mode and dy > 1: continue 
                if score > 60:
                    if dy == 0: score += YEAR_BONUS_EXACT
                    elif dy == 1: score += YEAR_BONUS_CLOSE
                if not strict_mode and dy > 1 and score < 92: score -= min(dy, 15)

            if db_author and fuzz.partial_ratio(normalize_for_fuzz(db_author), norm_query_fuzz) >= 90: score += AUTHOR_BONUS
            if query_place and db_place and fuzz.partial_ratio(normalize_for_fuzz(query_place), normalize_for_fuzz(db_place)) > 85: score += PLACE_BONUS

            if score >= MIN_ACCEPT_SCORE:
                out.append({"vd_id": vd_id, "vd_id_norm": normalize_db_vd_id(vd_id), "score": min(score, 100), "source": "RUN_1"})
        return out

    candidates = []
    if strict_q: candidates.extend(process(sql_search(strict_q, query_year), True))
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    if (not candidates or candidates[0]["score"] < 95) and broad_q:
        cands_broad = process(sql_search(broad_q, None), False)
        existing = {c["vd_id_norm"] for c in candidates}
        for c in cands_broad:
            if c["vd_id_norm"] not in existing: candidates.append(c)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

# Run 2
def execute_run_2(conn, broad_q, query_year, norm_query_fuzz, query_author):
    if not broad_q or not query_year or not query_author: return []
    c = conn.cursor()
    y = int(query_year)
    sql = "SELECT clean_author, clean_title, year, place, vd_id FROM vd18 WHERE vd18 MATCH ? AND year >= ? AND year <= ? LIMIT 300"
    rows = c.execute(sql, (broad_q, str(y-3), str(y+3))).fetchall()
    
    out = []
    norm_q_auth = normalize_for_fuzz(query_author)
    if len(norm_q_auth) < 4: return [] 

    for db_author, db_title, db_year, db_place, vd_id in rows:
        if not vd_id:
            continue
        if db_author: continue 
        full_label = f"{db_title or ''}".strip()
        norm_db_label = normalize_for_fuzz(full_label)
        if fuzz.partial_ratio(norm_q_auth, norm_db_label) > 90:
            score = fuzz.WRatio(norm_query_fuzz, norm_db_label) + 30
            dy = abs(y - int(db_year))
            if dy > 0: score -= dy 
            if score >= MIN_ACCEPT_SCORE:
                out.append({"vd_id": vd_id, "vd_id_norm": normalize_db_vd_id(vd_id), "score": min(score, 100), "source": "RUN_2"})
    return out

# Run 3: Entity fallback
def execute_run_3(conn, broad_q, norm_query_fuzz, raw_query):
    if not broad_q: return []
    c = conn.cursor()
    sql = "SELECT clean_author, clean_title, year, place, vd_id FROM vd18 WHERE vd18 MATCH ? LIMIT 200"
    rows = c.execute(sql, (broad_q,)).fetchall()
    
    entities = [w for w in re.findall(r'\b[A-Z][a-z]{3,}\b', raw_query) if w.lower() not in STOPWORDS]
    
    out = []
    for db_author, db_title, db_year, db_place, vd_id in rows:
        if not vd_id:
            continue
        full_label = f"{db_author or ''} {db_title or ''}".strip()
        norm_db_label = normalize_for_fuzz(full_label)
        score = fuzz.token_set_ratio(norm_query_fuzz, norm_db_label)
        
        hits = 0
        for ent in entities:
             if ent.lower() in norm_db_label:
                 hits += 1
        score += (hits * ENTITY_BONUS)

        if vd_id.upper().startswith("VD18"):
            score += VD18_PREF_BONUS

        if score >= MIN_ACCEPT_SCORE:
            out.append({"vd_id": vd_id, "vd_id_norm": normalize_db_vd_id(vd_id), "score": min(score, 100), "source": "RUN_3"})
            
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# Run 4: Rare terms
def execute_run_4(conn, norm_query_fts, norm_query_fuzz):
    """
    Finds the longest/rarest words in the query and searches specifically for them.
    Solves cases like "Eine neue Entrevue" -> Finds "Entrevue".
    """
    tokens = [t for t in norm_query_fts.split() if len(t) >= 5 and t not in STOPWORDS]
    if not tokens: return []
    
    tokens.sort(key=len, reverse=True)
    top_terms = tokens[:3] # Take top 3 longest words
    
    fts_query = " OR ".join(top_terms)
    
    c = conn.cursor()
    sql = "SELECT clean_author, clean_title, year, place, vd_id FROM vd18 WHERE vd18 MATCH ? LIMIT 100"
    rows = c.execute(sql, (fts_query,)).fetchall()
    
    out = []
    for db_author, db_title, db_year, db_place, vd_id in rows:
        if not vd_id:
            continue
        full_label = f"{db_author or ''} {db_title or ''}".strip()
        norm_db_label = normalize_for_fuzz(full_label)
        
        score = fuzz.token_set_ratio(norm_query_fuzz, norm_db_label)
        
        if score >= 65:
            out.append({"vd_id": vd_id, "vd_id_norm": normalize_db_vd_id(vd_id), "score": min(score, 100), "source": "RUN_4"})

    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# LLM judgmentT
def execute_llm_judgment(conn, query_str: str, top_candidates: list) -> str | None:
    if not top_candidates:
        return None

    candidate_ids = [c["vd_id"] for c in top_candidates]
    placeholders = ",".join("?" * len(candidate_ids))
    
    c = conn.cursor()
    sql = f"SELECT vd_id, clean_author, clean_title, year, place FROM vd18 WHERE vd_id IN ({placeholders})"
    rows = c.execute(sql, candidate_ids).fetchall()
    
    candidates_text = ""
    for r in rows:
        candidates_text += f"""
RECORD_ID: {r[0]}
AUTHOR:    {r[1]}
TITLE:     {r[2]}
YEAR:      {r[3]}
PLACE:     {r[4]}
-----------------------
"""

    prompt = f"""
    You are a bibliographic expert. Match the query to the best database record.

    SEARCH QUERY: "{query_str}"
    
    CANDIDATES:
    {candidates_text}
    
    INSTRUCTIONS:
    1. Identify the record that best matches the search query.
    2. PREFERENCE RULE: If two records are similar, ALWAYS prefer the one with a 'vd18' prefix in the RECORD_ID over 'vd17'.
    3. Be flexible with OCR errors (e.g., 'Bot' = 'Botschaft') and partial title matches.
    4. If the query starts with 'Eine neue...' and the candidate says 'Besondere...' (or similar variations), accept it if the core nouns match.
    5. If the query has no year, find the best text match regardless of year, but still prefer 'vd18' IDs.
    6. Return the RECORD_ID of the best match. If nothing matches, return "NONE".
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", 
                response_schema=MatchDecision,
                temperature=0.0 
            )
        )
        result = json.loads(response.text)
        winner_id = result.get("best_match_vd_id", "NONE")
        
        if winner_id != "NONE" and any(c["vd_id"] == winner_id for c in top_candidates):
            return winner_id
            
    except Exception as e:
        print(f"LLM Error: {e}")
        return None
        
    return None

# Aggregation
def get_search_results(query: str, db_path: Path, query_place: str, query_author: str):
    query_year = extract_query_year(query)
    norm_query_fts = normalize_for_fts(query)
    norm_query_fuzz = normalize_for_fuzz(query)
    strict_q, broad_q = build_fts_queries(norm_query_fts)
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        r1 = execute_run_1(conn, norm_query_fts, strict_q, broad_q, query_year, norm_query_fuzz, query_place, query_author)
        r2 = execute_run_2(conn, broad_q, query_year, norm_query_fuzz, query_author)
        r3 = execute_run_3(conn, broad_q, norm_query_fuzz, query)
        r5 = execute_run_4(conn, norm_query_fts, norm_query_fuzz)

        final_map = {}
        all_results = r1 + r2 + r3 + r5
        
        exclude_vd17 = False
        if query_year and query_year.isdigit() and int(query_year) > 1700:
            exclude_vd17 = True
            
        for r in all_results:
            vid = r["vd_id_norm"]
            if exclude_vd17 and vid.startswith("vd17"):
                continue

            if vid not in final_map:
                final_map[vid] = r
            else:
                if r["score"] > final_map[vid]["score"]:
                    final_map[vid] = r
        
        final_list = list(final_map.values())
        final_list.sort(key=lambda x: x["score"], reverse=True)

        llm_winner_id = "-"
        if final_list:
            top_cands = final_list[:25] # Wide net
            
            winner = execute_llm_judgment(conn, query, top_cands)
            
            if winner:
                llm_winner_id = winner
                for item in final_list:
                    if item["vd_id"] == winner:
                        item["score"] = 100
                        item["source"] += "+LLM"
                        final_list.remove(item)
                        final_list.insert(0, item)
                        break

        return final_list, r1, r2, r3, r5, llm_winner_id
        
    finally:
        conn.close()

if __name__ == "__main__":
    print(f" my{GOOGLE_API_KEY}")
    db_path = INDEX_DIR / DB_NAME
    db_sanity_check(db_path)

    if not TEST_SET_PATH.exists():
        print("Test set missing.")
    else:
        with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        stats = {"total": 0, "found": 0}
        
        print(f"{'STATUS':<8} | {'QUERY':<30} | {'TARGET ID':<15} | {'RUN1':<10} | {'RUN2':<10} | {'LLM':<10} | {'FINAL'}")
        print("-" * 135)
        
        for entry in data:
            bibl = entry.get("json_representation", {}).get("item", {}).get("bibl", [])
            bibl_items = [bibl] if isinstance(bibl, dict) else bibl
            gt_items = entry.get("VD18 request", [])
            gt_items = [gt_items] if isinstance(gt_items, dict) else gt_items

            for b, gt in zip(bibl_items, gt_items):
                stats["total"] += 1
                q, q_place, q_author = build_query_string(b)
                target = extract_vd18_id_from_gt(str(gt.get("VD18_ID", "")))
                
                if not target: continue
                
                final_res, r1, r2, r3, r5, llm_id = get_search_results(q, db_path, q_place, q_author)
                
                r1_id = r1[0]['vd_id'] if r1 else "-"
                r5_id = r5[0]['vd_id'] if r5 else "-"
                final_id = final_res[0]['vd_id'] if final_res else "-"
                
                hit = any(r["vd_id_norm"] == target for r in final_res[:5])
                status = "✅" if hit else "❌"
                if hit: stats["found"] += 1
                
                target_display = str(gt.get("VD18_ID", "")).replace(" ", "")
                r1_dsp = (r1_id[:10] + '..') if len(r1_id) > 10 else r1_id
                r5_dsp = (r5_id[:10] + '..') if len(r5_id) > 10 else r5_id
                llm_dsp = (llm_id[:10] + '..') if len(llm_id) > 10 else llm_id

                print(f"{status:<8} | {q[:30]:<30} | {target_display:<15} | {r1_dsp:<10} | {r5_dsp:<10} | {llm_dsp:<10} | {final_id}")

        print(f"\nScore: {stats['found']}/{stats['total']}")
