"""Pre-compute Qwen keyword extractions for all session×turn pairs.

Saves a JSON cache keyed by MD5(category_specificity_retrieval_input).
At inference time, HYBRID_MODEL loads this cache and skips Qwen entirely.

Usage:
    python -m mcrs.preprocess.preprocess_keywords \
        --model Qwen/Qwen3-8B \
        --output precomputed/keyword_cache.json
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


def _cache_key(category: str, specificity: str, retrieval_input: str) -> str:
    """Cache key must match HYBRID_MODEL._lookup_or_extract() exactly.

    retriever.py uses: md5(f"{category}_{specificity}_{query}")
    """
    raw = f"{category}_{specificity}_{retrieval_input}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _serializable(result: dict) -> dict:
    """Keep only JSON-serializable fields from extract_structured_query result."""
    out = {}
    for key, val in result.items():
        if val is None or isinstance(val, (str, int, float, bool, list, dict)):
            out[key] = val
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="Qwen/Qwen3-8B")
    parser.add_argument("--output",  default="precomputed/keyword_cache.json")
    parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--item_db", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--split",   default="test")
    # category/specificity 필드명은 데이터셋 컬럼명에 따라 다를 수 있음
    parser.add_argument("--cat_field",  default="category",
                        help="데이터셋에서 category를 담은 컬럼명")
    parser.add_argument("--spec_field", default="specificity",
                        help="데이터셋에서 specificity를 담은 컬럼명")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen from {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device).eval()
    lm_components = (model, tokenizer, device)

    print("Loading item DB...")
    item_db = MusicCatalogDB(args.item_db, ["all_tracks"], CORPUS_TYPES)

    print(f"Loading dataset {args.dataset} split={args.split}...")
    db = load_dataset(args.dataset, split=args.split)

    # run_inference_devset.py와 동일한 방식으로 category/specificity 읽기:
    # item['conversation_goal']['category'] / item['conversation_goal']['specificity']
    sample_item = db[0]
    has_conv_goal = "conversation_goal" in sample_item
    if has_conv_goal:
        print("[info] Reading category/specificity from item['conversation_goal'].")
    else:
        print(f"[warn] 'conversation_goal' field not found. "
              f"Available fields: {list(sample_item.keys())}")
        print("[warn] Will use category=None, specificity=None (default prompt).")

    # Build all (key, retrieval_input, category, specificity) tuples, skip duplicates
    entries = []
    seen_keys = set()
    for item in db:
        if has_conv_goal:
            goal        = item["conversation_goal"]
            category    = goal.get("category")
            specificity = goal.get("specificity")
        else:
            category    = None
            specificity = None

        for turn_number in range(1, 9):
            try:
                chat_history, user_query = _chat_history_parser(
                    item["conversations"], item_db, turn_number
                )
            except (IndexError, KeyError):
                # 턴 수가 8 미만인 세션은 조용히 건너뜀
                break

            retrieval_input = _build_retrieval_input(chat_history, user_query)
            key = _cache_key(category or "", specificity or "", retrieval_input)

            if key not in seen_keys:
                seen_keys.add(key)
                entries.append((key, retrieval_input, category, specificity))

    print(f"Total unique (category, specificity, query) pairs: {len(entries)}")

    # 기존 캐시 파일이 있으면 로드해서 이어쓰기 (재시작 복구용)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[resume] Loaded existing cache: {len(cache)} entries. "
              f"Skipping already-computed keys.")
    else:
        cache = {}

    skipped = 0
    for key, retrieval_input, category, specificity in tqdm(entries, desc="Extracting"):
        # 이미 캐시에 있으면 스킵 (중단 후 재시작 시 처음부터 다시 안 해도 됨)
        if key in cache:
            skipped += 1
            continue

        result = extract_structured_query(
            retrieval_input,
            lm_components,
            category=category,
            specificity=specificity,
        )
        cache[key] = _serializable(result)

    print(f"Skipped (already cached): {skipped}")
    print(f"Total cache entries now: {len(cache)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
