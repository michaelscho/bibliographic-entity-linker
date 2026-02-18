# Bibliographic Entity Linking (VD16/VD17/VD18)

This repository contains a **hybrid Retrieve and Rank pipeline** designed to link noisy, OCR-degraded bibliographic strings from historical advertisements, such as the **Avisblatt**, to canonical records in the VD16, VD17, and VD18 directories (Verzeichnis der im deutschen Sprachraum erschienenen Drucke). It is meant as a proof of concept in the context of **Avisblatt** annotation and may need further adaption on other corpora.

## Context and Motivation
This tool was developed as a proof of concept for the annotation of the **Basler Avisblatt (1729–1844)**, a significant source for the early modern book market. As described in our research on scalable entity detection in 18th-century newspaper advertisements, the *Avisblatt* presents unique challenges:
* **High Variance:** Advertisements range from professional publisher announcements to private sales and auctions.
* **Data Quality:** Despite good OCR (CER < 1%), errors like character confusion (e.g., *Basel* vs. *Bafel*) and segmentation issues persist.
* **Bibliographic Ambiguity:** Entries often lack standardized formatting, use abbreviated titles, or group multiple works in single blocks of text.

Standard RAG (Retrieval-Augmented Generation) systems often fail on this task because vector embeddings struggle with specific character-level noise or miss semantic context. This project solves this using a **deterministic multi-stage retrieval** followed by an **LLM-based reasoning step**, to map these non-standard, noisy references to the structured authority data of the VD18, serving as a downstream task following Named Entity Recognition (NER).

## Architecture

The system operates in two distinct phases:

### Phase 1: Candidate Generation
We utilize a local **SQLite FTS5 database** with **Trigram Tokenization**. Trigrams break words into 3-character overlapping sequences, making the index highly resilient to OCR errors. To ensure relevant candidates are found, the pipeline aggregates results from four complementary search strategies ("Runs"):

1.  **Run A (Fuzzy Baseline):** Boolean FTS search combined with rapid fuzzy string matching (Levenshtein). Prioritizes matches where the year aligns.
2.  **Run B (Temporal Focus):** Aggressive search within a narrow time window (`Year += 3`). Matches records even if title/author similarity is low, catching severe OCR degradation.
3.  **Run C (Entity Boosting):** Extracts capitalized terms (potential named entities) from the query to boost relevance for generic titles.
4.  **Run D (Rare Term Heuristic):** Isolates the longest, most distinctive words in the query to bypass noise in common stopwords.

### Phase 2: Semantic Re-Ranking (High Precision)
The top candidates (Top-N) are sent to a Large Language Model (**Gemini 2.5 Pro** or similar). The LLM acts as a reasoning engine to:
* **Expand Abbreviations:** Maps `Hist.` to `Historie`, etc.
* **Resolve Ambiguity:** Decides between multiple editions based on textual evidence when the date is missing or conflicting.
* **Apply Preference Rules:** Prioritizes VD18 IDs over VD17 when applicable.

![Architecture](./bel-architecture.png)

## Data Source and Acknowledgments

The core metadata used to build the index is provided by the **Staatsbibliothek zu Berlin - Berlin State Library**. We gratefully acknowledge their work in curating and publishing the *Verzeichnis der im deutschen Sprachraum erschienenen Drucke*.

If you use this pipeline or the underlying data, please cite the original dataset:

> **Federbusch, M., Stachowiak, R., & Lehmann, J. (2025).** *Metadata of the "Verzeichnis der im deutschen Sprachraum erschienen Drucke"*. Staatsbibliothek zu Berlin. https://doi.org/10.5281/zenodo.15167939

```bibtex
@dataset{federbusch_2025_15167939,
  author       = {Federbusch, Maria and Stachowiak, Remigiusz and Lehmann, Jörg},
  title        = {Metadata of the "Verzeichnis der im deutschen Sprachraum erschienen Drucke"},
  month        = apr,
  year         = 2025,
  publisher    = {Staatsbibliothek zu Berlin - Berlin State Library},
  doi          = {10.5281/zenodo.15167939},
  url          = {[https://doi.org/10.5281/zenodo.15167939](https://doi.org/10.5281/zenodo.15167939)}
}
```

## Repository Structure

| File | Description |
| :--- | :--- |
| `download_vd_data.py` | ETL script. Downloads `SBB/VD-Metadata` from Hugging Face, cleans metadata, and exports to JSONL. |
| `create_el_index.py` | Indexer. Ingests JSONL and builds the optimized SQLite FTS5 database (`vd18_fts.db`). |
| `evaluation_unique.py` | Runs the pipeline expecting a single best match (Precision/Recall metrics). |
| `evaluation_multiple.py` | Runs the pipeline returning a ranked list of plausible candidates for ambiguous queries. |
| `test_set.json` | A curated dataset of difficult historical queries with Ground Truth VD IDs. |

## Installation and Requirements

The pipeline requires Python 3.9+ and the following dependencies:

```bash
pip install rapidfuzz google-generativeai datasets tqdm
```

*Note: `sqlite3` is required but is included in the Python standard library.*

## Usage

### 1. Data Preparation
Download the dataset and prepare the JSONL file.
```bash
python download_vd_data.py
```
*Output: `data/register/vd_works.jsonl`*

### 2. Indexing
Build the local inverted index. This process normalizes text (NFKD, lowercase, punctuation removal) and creates the trigram tokens.
```bash
python create_el_index.py
```
*Output: `data/indices/vd18_fts.db`*

### 3. Evaluation
Run the matching pipeline against the test set. Ensure you have your API key set. Change code to adapt to other LLM provider.

```bash
export GOOGLE_API_KEY="your_key_here"
python evaluation_unique.py
```

## Configuration

You can adjust the pipeline behavior via environment variables or by modifying the config section in the scripts:

* **`GOOGLE_API_KEY`**: (Required) API key for the Gemini LLM re-ranker.
* **`VD18_DB_NAME`**: (Optional) Override the database filename (default: `vd18_fts.db`).

## Methodological Notes

* **Normalization:** All text is normalized to NFKD form, stripped of combining characters, lowercased, and cleaned of non-alphanumeric characters before indexing.
* **VD18 Preference:** The logic explicitly prefers `vd18` prefixed IDs over `vd17` or `vd16` if the text match score is comparable, reflecting the project's focus on 18th-century prints.
* **Date Handling:** While the system heavily penalizes date mismatches in strict mode, specific "fallback runs" allow for date discrepancies (e.g., matching a 1748 edition when the query asks for 1732) if the textual evidence is overwhelming.

## Citation
If you use this tool, please cite apropriatly.

```bibtex

```