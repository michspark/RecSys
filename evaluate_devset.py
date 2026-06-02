"""
Evaluation script for music recommendation systems.

This script evaluates recommendation system predictions against ground truth data
from the TalkPlayData-2 dataset, computing various metrics across conversation turns.
Outputs overall scores + per-category + per-specificity + per-(category, specificity) breakdowns.
"""

import os
import json
from typing import List, Dict, Any
from datasets import load_dataset
from metrics import compute_recsys_metrics, compute_lexical_diversity, compute_catalog_diversity
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Evaluate music recommendation system predictions")
parser.add_argument("--tid", type=str, default="llama1b_bm25",
                    help="Name of the experiment (used to locate prediction files)")
parser.add_argument("--eval_dataset", type=str, default="devset")
args = parser.parse_args()


def df_filtering(df, session_id, turn_number):
    session_filter = df['session_id'] == session_id
    turn_number_filter = df['turn_number'] == turn_number
    return df[session_filter & turn_number_filter].iloc[0]


def compute_gt_rank(predicted_track_ids: List[str], ground_truth_id: str) -> int:
    """GT 트랙이 예측 리스트에서 몇 번째에 있는지 반환 (1-indexed).

    리스트에 없으면 len(predicted_track_ids) + 1 반환.
    """
    try:
        return predicted_track_ids.index(ground_truth_id) + 1
    except ValueError:
        return len(predicted_track_ids) + 1


def aggregate_group(rows: List[Dict]) -> Dict:
    """row 리스트에서 ndcg 평균과 gt_rank 평균을 계산."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in ("session_id", "turn_number", "category", "specificity")]
    return {
        k: round(sum(r[k] for r in rows) / len(rows), 6)
        for k in keys
    }


def main(args) -> None:
    ground_truth = json.load(open("exp/ground_truth/devset.json", "r"))
    predictions  = json.load(open(f"exp/inference/devset/{args.tid}.json", "r"))
    df_predictions  = pd.DataFrame(predictions)
    df_ground_truth = pd.DataFrame(ground_truth)

    # 데이터셋에서 session_id → (category, specificity) 매핑 로드
    print("Loading dataset for category/specificity labels...")
    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    session_to_cat: Dict[str, str] = {}
    session_to_spec: Dict[str, str] = {}
    for item in db:
        sid  = item["session_id"]
        goal = item.get("conversation_goal", {})
        session_to_cat[sid]  = goal.get("category", "unknown")
        session_to_spec[sid] = goal.get("specificity", "unknown")

    list_of_recommended_track_ids = []
    list_of_responses             = []
    results: List[Dict]           = []

    for _, row in tqdm(df_ground_truth.iterrows(), total=len(df_ground_truth)):
        session_id  = row["session_id"]
        turn_number = row["turn_number"]

        pred = df_filtering(df_predictions, session_id, turn_number)
        gt   = df_filtering(df_ground_truth, session_id, turn_number)

        predicted_ids    = pred["predicted_track_ids"]
        gt_id            = gt["ground_truth_track_id"]
        recsys_metrics   = compute_recsys_metrics(predicted_ids, [gt_id], [1, 10, 20])
        gt_rank          = compute_gt_rank(predicted_ids, gt_id)

        list_of_recommended_track_ids.extend(predicted_ids)
        list_of_responses.append(pred["predicted_response"])

        results.append({
            "session_id":  session_id,
            "turn_number": turn_number,
            "category":    session_to_cat.get(session_id, "unknown"),
            "specificity": session_to_spec.get(session_id, "unknown"),
            "gt_rank":     gt_rank,
            **recsys_metrics,
        })

    # ── 전체 평균 (기존 방식 유지) ─────────────────────────────────────────────
    df_results          = pd.DataFrame(results)
    df_turn_wise        = df_results.drop(columns=["session_id", "category", "specificity"]).groupby("turn_number").mean()
    overall             = df_turn_wise.mean(axis=0).to_dict()

    music_catalog       = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    total_catalog_size  = len(music_catalog)
    overall["catalog_diversity"] = compute_catalog_diversity(list_of_recommended_track_ids, total_catalog_size)
    overall["lexical_diversity"] = compute_lexical_diversity(list_of_responses)
    overall["total_catalog_size"] = total_catalog_size

    # ── category별 ────────────────────────────────────────────────────────────
    by_category: Dict[str, Dict] = {}
    for cat, group in df_results.groupby("category"):
        rows = group.drop(columns=["session_id", "category", "specificity"]).to_dict(orient="records")
        agg  = {k: round(float(group[k].mean()), 6) for k in ["gt_rank", "ndcg@1", "ndcg@10", "ndcg@20"]}
        agg["count"] = len(group["session_id"].unique())
        by_category[cat] = agg

    # ── specificity별 ─────────────────────────────────────────────────────────
    by_specificity: Dict[str, Dict] = {}
    for spec, group in df_results.groupby("specificity"):
        agg  = {k: round(float(group[k].mean()), 6) for k in ["gt_rank", "ndcg@1", "ndcg@10", "ndcg@20"]}
        agg["count"] = len(group["session_id"].unique())
        by_specificity[spec] = agg

    # ── (category, specificity) 조합별 ────────────────────────────────────────
    by_cat_spec: Dict[str, Dict] = {}
    for (cat, spec), group in df_results.groupby(["category", "specificity"]):
        key  = f"{cat}_{spec}"
        agg  = {k: round(float(group[k].mean()), 6) for k in ["gt_rank", "ndcg@1", "ndcg@10", "ndcg@20"]}
        agg["count"] = len(group["session_id"].unique())
        by_cat_spec[key] = agg

    # ── 저장 ──────────────────────────────────────────────────────────────────
    output = {
        "overall":                overall,
        "by_category":            by_category,
        "by_specificity":         by_specificity,
        "by_category_specificity": by_cat_spec,
    }

    os.makedirs("exp/scores/devset", exist_ok=True)
    out_path = f"exp/scores/devset/{args.tid}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out_path}")

    # 터미널에 요약 출력
    print(f"\n── Overall ──")
    for k in ["ndcg@1", "ndcg@10", "ndcg@20", "catalog_diversity", "lexical_diversity"]:
        print(f"  {k}: {overall[k]:.4f}")
    print(f"\n── By Category (ndcg@10) ──")
    for cat in sorted(by_category):
        print(f"  {cat}: {by_category[cat]['ndcg@10']:.4f}  (gt_rank avg={by_category[cat]['gt_rank']:.1f}, n={by_category[cat]['count']})")
    print(f"\n── By Specificity (ndcg@10) ──")
    for spec in sorted(by_specificity):
        print(f"  {spec}: {by_specificity[spec]['ndcg@10']:.4f}  (gt_rank avg={by_specificity[spec]['gt_rank']:.1f}, n={by_specificity[spec]['count']})")


if __name__ == "__main__":
    main(args)
