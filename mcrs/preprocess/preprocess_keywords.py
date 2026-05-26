"""Pre-compute Qwen keyword extractions for all session×turn pairs.

Saves a JSON cache keyed by MD5 hash of each retrieval_input string.
At inference time, HYBRID_MODEL loads this cache and skips Qwen entirely.

Usage:
    python -m mcrs.preprocess.preprocess_keywords \
        --model Qwen/Qwen3-4B \
        --output cache/keyword_cache.json
"""

import os
import sys
import json
import hashlib
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mcrs.db_item import MusicCatalogDB
from mcrs.retrieval_modules.extract_query_comp import extract_structured_query


CORPUS_TYPES = ["track_name", "artist_name", "album_name", "release_date", "tag_list"]


def _chat_history_parser(conversations, item_db, target_turn_number):
    """Build chat_history and user_query at target_turn — same logic as run_inference_devset.py."""
    df = pd.DataFrame(conversations)
    df_history = df[df["turn_number"] < target_turn_number]
    chat_history = []
    for row in df_history.to_dict(orient="records"):
        role = row["role"]
        content = row["content"]
        if role == "music":
            role = "assistant"
            content = item_db.id_to_metadata(content)
        chat_history.append({"role": role, "content": content})
    user_query = df[df["turn_number"] == target_turn_number].iloc[0]["content"]
    return chat_history, user_query


def _build_retrieval_input(chat_history, user_query):
    """Build retrieval_input string — same logic as crs_baseline.batch_chat."""
    session_memory = chat_history + [{"role": "user", "content": user_query}]
    return "\n".join(f"{m['role']}: {m['content']}" for m in session_memory)


def query_hash(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output", default="precomputed/keyword_cache.json")
    parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--item_db", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen from {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    lm_components = (model, tokenizer, device)

    print("Loading item DB...")
    item_db = MusicCatalogDB(args.item_db, ["all_tracks"], CORPUS_TYPES)

    print(f"Loading dataset {args.dataset} split={args.split}...")
    db = load_dataset(args.dataset, split=args.split)

    # Build all (retrieval_input, key) pairs first to detect duplicates
    entries = []
    seen_keys = set()
    for item in db:
        for turn_number in range(1, 9):
            chat_history, user_query = _chat_history_parser(item["conversations"], item_db, turn_number)
            retrieval_input = _build_retrieval_input(chat_history, user_query)
            key = query_hash(retrieval_input)
            if key not in seen_keys:
                seen_keys.add(key)
                entries.append((key, retrieval_input))

    print(f"Total unique queries to process: {len(entries)}")

    cache = {}
    for key, retrieval_input in tqdm(entries, desc="Extracting keywords"):
        components = extract_structured_query(retrieval_input, lm_components)
        # Store only serializable fields (tensors already converted in extract_structured_query)
        cache[key] = {
            "direct_request": components["direct_request"],
            "bge_query": components["bge_query"],
            "clap_keywords": components["clap_keywords"],
            "rejected": components["rejected"],
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"Saved {len(cache)} entries → {args.output}")


if __name__ == "__main__":
    main()
