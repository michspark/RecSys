# RecSys Challenge 2026 — Music CRS

A conversational music recommender for the **RecSys Challenge 2026 Music CRS** track.
Given a multi-turn conversation, the system retrieves a ranked list of tracks and generates a
natural-language response that recommends the top track.

The upstream challenge description (dataset schemas, inference JSON format, metric definitions)
lives in [`docs/challenge_spec.md`](docs/challenge_spec.md).

## Pipeline

```
conversation ──→ retrieval_input (cumulative query)
                        │
                        ▼
      Retriever (config: retrieval_type) ──→ top-20 track IDs   → nDCG, catalog diversity
                        │
                        ▼
      top-1 metadata + history ──→ LM ──→ response              → lexical diversity
```

`mcrs.CRS_BASELINE` (`mcrs/crs_baseline.py`) wires together the retriever, the LM and the
track/user metadata DBs.

### Retrievers (`mcrs/retrieval_modules/`)

| `retrieval_type`      | Description |
|-----------------------|-------------|
| `bm25`                | Lexical BM25 over track metadata |
| `bert` / `bge`        | Dense retrieval with BERT / BGE embeddings |
| `hybrid`              | BGE + CLAP with LLM-extracted structured queries and a CF-BPR reranker |
| `anchor_cf`           | BGE cumulative query, blended with the previously accepted track ("anchor") and fused with CF-BPR scores |
| `anchor_bge_bm25_cf`  | `anchor_cf` dense channel + BM25 sparse channel, combined with Reciprocal Rank Fusion (**best**) |

### Response generation (`mcrs/lm_modules/`)

- `v1` — Llama-3.2-1B-Instruct baseline.
- `v2` — any HF chat model (e.g. Qwen3-8B) with an instruction-style prompt, sampling and a
  "natural" system prompt; also exposes a pointwise Yes/No scorer used by the LLM reranker.
- `api` — OpenAI Chat Completions backend.

Optional extras: an LLM reranker (`mcrs/rerank/`) and a catalog-diversity post-processor
(`mcrs/postprocess/`), both toggled from the config.

## Setup

```bash
uv venv .venv --python=3.10 && source .venv/bin/activate
uv pip install -e .
uv pip install -r requirments.txt
```

## Usage

Each run is driven by a YAML file in `config/`; `--tid` is the filename without `.yaml`.

```bash
# 1. Build the ground truth (once)
python make_ground_truth.py
# → exp/ground_truth/devset.json

# 2. Run inference
python run_inference_devset.py --tid llama1b_anchor_bge_bm25_cf_devset --batch_size 16
# → exp/inference/devset/<tid>_<MMDD_HHMM>.json

# 3. Evaluate
python evaluate_devset.py --tid <tid>_<MMDD_HHMM> --eval_dataset devset
# → exp/scores/devset/<tid>_<MMDD_HHMM>.json

# Blind set submission
python run_inference_blindset.py --tid llama1b_anchor_bge_bm25_cf_blindset_A --eval_dataset blindset_A
```

The `hybrid` retriever reads a precomputed LLM keyword cache. Build it once with:

```bash
python -m mcrs.preprocess.preprocess_keywords --model Qwen/Qwen3-8B --output precomputed/keyword_cache.json
```

## Devset results

| Run                                 | nDCG@1 | nDCG@10 | nDCG@20 | Catalog div. | Lexical div. |
|-------------------------------------|-------:|--------:|--------:|-------------:|-------------:|
| Random (`lowerbound/`)              | 0.0000 | 0.0001 | 0.0001 | 0.965 | — |
| Popularity (`lowerbound/`)          | 0.0005 | 0.0018 | 0.0024 | 0.000 | — |
| `llama1b_bert_devset`               | 0.0018 | 0.0048 | 0.0063 | 0.061 | 0.205 |
| `llama1b_bge_devset`                | 0.0124 | 0.0692 | 0.0844 | 0.285 | 0.258 |
| `llama1b_anchor_bge_bm25_cf_devset` | 0.0479 | 0.1264 | 0.1468 | 0.509 | 0.259 |

## Repository layout

```
config/               Run configs (one YAML per --tid)
mcrs/
  crs_baseline.py     Pipeline: retrieval → (rerank) → response generation
  retrieval_modules/  BM25 / BERT / BGE / CLAP / SigLIP / hybrid / anchor retrievers
    extraction_prompts/  Per-category (A–K) query-extraction prompts for the hybrid retriever
  lm_modules/         Response-generation backends
  rerank/             LLM reranker
  postprocess/        Catalog-diversity post-processing
  preprocess/         Offline keyword-cache builder
  db_item/, db_user/  Track / user metadata accessors
  system_prompts/     System prompt templates
metrics/              nDCG, catalog and lexical diversity
lowerbound/           Random and popularity baselines
tips/                 Improvement notes from the challenge organizers
docs/                 Upstream challenge specification
```
