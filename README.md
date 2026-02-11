# Bibliographic Entity Linking (VD16/VD17/VD18)

This repository builds a local full‑text index from VD metadata and evaluates an entity‑linking pipeline that retrieves bibliographic record IDs (VD16/VD17/VD18) from noisy queries.

It is intended to help link OCR‑like bibliographic strings to canonical VD records, with a focus on VD18 and a fallback to VD17 when appropriate.

## What’s in this repo

- `download_vd_data.py` downloads the VD metadata dataset from Hugging Face and writes it as JSONL.
- `create_el_index.py` builds a local SQLite FTS5 index from the JSONL.
- `evaluation_unique.py` evaluates a single‑best‑match pipeline.
- `evaluation_multiple.py` evaluates a multi‑match pipeline for ambiguous cases.
- `test_set.json` contains evaluation queries and targets.

## Data flow

1. Download the dataset into `data/register/vd_works.jsonl`.
2. Build the FTS index in `data/indices/vd18_fts_2.db`.
3. Run evaluation against `test_set.json`.

## Indexing method and rationale

The index is built using **SQLite FTS5** with the **trigram tokenizer**.

Why FTS5 + trigram:
- The source text is noisy and often contains OCR errors or archaic spelling.
- Trigram tokenization is robust to minor spelling variation and character‑level noise.
- SQLite FTS5 provides fast, local, zero‑dependency full‑text search and BM25 ranking.

What is indexed:
- `search_text` contains concatenated, normalized fields: author, title, year, and place.
- `clean_author` and `clean_title` are stored for downstream scoring.
- `year`, `place`, and `vd_id` are stored as unindexed fields for filtering and output.

Normalization:
- Unicode is normalized to NFKD.
- Diacritics are stripped.
- Punctuation is removed.
- Text is lower‑cased and whitespace‑collapsed.

Only records with a VD identifier are indexed. This avoids polluting search results with rows that cannot be linked.

## Retrieval methods used in evaluation

Both evaluation scripts use a multi‑stage retrieval strategy:

1. **FTS strict query**
   - Builds a query from the longest non‑stopword tokens.
   - Uses `AND` between the top tokens.
   - Filters by year if present.
2. **FTS broad query**
   - Falls back to a broader `OR` query if strict results are weak.
3. **Fuzzy re‑ranking**
   - Uses RapidFuzz token‑set and partial ratios on normalized text.
   - Adds bonuses for matching year, author, and place.
4. **Specialized fallback runs**
   - Hidden‑author recovery with relaxed year window.
   - Entity‑keyword emphasis for capitalized terms.
   - Rare‑term search for long, distinctive tokens.
5. **LLM tie‑breaker**
   - Gemini is used as a last‑step judge to select the best ID among top candidates.

The pipeline prefers VD18 IDs when ambiguous and can exclude VD17 for post‑1700 queries.

## Usage

```bash
python download_vd_data.py
python create_el_index.py
python evaluation_unique.py
python evaluation_multiple.py
```

### Environment variables

- `GOOGLE_API_KEY` enables the LLM selection step.
- `VD18_DB_NAME` overrides the default index filename (`vd18_fts.db`).

## Notes

If you created the index from the Hugging Face JSONL, make sure the indexer ran with the current `create_el_index.py`. Earlier versions could write many rows with empty `vd_id`, which breaks evaluation.
