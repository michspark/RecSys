"""
추론 결과(predicted_track_ids)를 (category × specificity) 셀별 Recall@20으로 채점하는 진단 스크립트.

evaluate_devset.py 와 동일한 소스(저장된 추론 JSON + ground truth)를 사용하되,
turn별 macro 평균 대신 (category, specificity) 셀별로 Recall@20을 쪼개어 표로 만든다.
모델을 로드하지 않으므로 빠르다.

핵심 성질: GT가 턴당 트랙 1개뿐이므로 Recall@20 = Hit@20 = (GT가 top-20 안에 있으면 1, 아니면 0).

사용법:
  python eval_recall_heatmap.py --pred exp/inference/devset/llama1b_bge_devset_0604_1638.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from datasets import load_dataset

# specificity 셀을 항상 같은 순서로 표에 배치하기 위한 고정 순서
SPECIFICITY_ORDER: List[str] = ["HH", "HL", "LH", "LL"]


def main(args: argparse.Namespace) -> None:
    """저장된 추론 JSON을 읽어 (category × specificity) Recall@20 표를 만들고 저장한다."""
    # ── 1. 추론 결과(예측) 로드 ────────────────────────────────────────────────
    # 실제 파이프라인이 저장한 top-20 = predicted_track_ids 를 그대로 채점한다.
    predictions = json.load(open(args.pred, "r", encoding="utf-8"))
    # (session_id, turn_number) -> predicted_track_ids 로 O(1) 조회용 dict 구성
    pred_by_key: Dict[tuple, List[str]] = {
        (row["session_id"], row["turn_number"]): row["predicted_track_ids"]
        for row in predictions
    }

    # ── 2. 정답(ground truth) 로드 ────────────────────────────────────────────
    ground_truth = json.load(open(args.gt, "r", encoding="utf-8"))

    # ── 3. category/specificity 라벨 로드 (세션 단위) ─────────────────────────
    # 라벨은 추론 JSON에 없으므로 원본 데이터셋의 conversation_goal에서 가져온다.
    print("[heatmap] Loading dataset for category/specificity labels...")
    db = load_dataset(args.dataset, split=args.split)
    session_to_label: Dict[str, Dict[str, str]] = {}
    for item in db:
        goal = item.get("conversation_goal", {})
        session_to_label[item["session_id"]] = {
            "category": goal.get("category", "?"),
            "specificity": goal.get("specificity", "?"),
        }

    # ── 4. 정답 행마다 예측과 매칭 → Recall@20(=Hit@20) 계산 ──────────────────
    records: List[Dict[str, Any]] = []
    missing_pred = 0  # 예측에 없는 (session, turn) 개수 추적
    for gt_row in ground_truth:
        session_id = gt_row["session_id"]
        turn_number = gt_row["turn_number"]
        gt_track_id = gt_row["ground_truth_track_id"]

        # 해당 (session, turn)의 예측 top-k 가져오기
        predicted_track_ids = pred_by_key.get((session_id, turn_number))
        if predicted_track_ids is None:
            missing_pred += 1
            continue  # 예측 파일에 없는 행은 건너뜀

        label = session_to_label.get(session_id)
        if label is None:
            continue  # 데이터셋에 없는 세션은 건너뜀

        # GT가 top-k 안에 있으면 hit=1 (단일 정답이라 Recall@k = Hit@k)
        top_ids = predicted_track_ids[: args.topk]
        hit = 1.0 if gt_track_id in top_ids else 0.0
        records.append(
            {
                "category": label["category"],
                "specificity": label["specificity"],
                "hit": hit,
            }
        )

    if missing_pred:
        print(f"[heatmap] WARNING: {missing_pred} ground-truth rows had no matching prediction.")

    # ── 5. 집계 + 출력 ────────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    # specificity 열 순서를 고정값 중 실제 존재하는 것만으로 정렬
    present_specs = [s for s in SPECIFICITY_ORDER if s in df["specificity"].unique()]

    # Recall@k 셀 표 (행=category, 열=specificity) = 셀별 hit 평균
    recall_pivot = df.pivot_table(
        index="category", columns="specificity", values="hit", aggfunc="mean"
    ).reindex(columns=present_specs)
    # 셀별 샘플 수 표 (소표본 셀 식별용)
    count_pivot = df.pivot_table(
        index="category", columns="specificity", values="hit", aggfunc="count"
    ).reindex(columns=present_specs)

    # 행/열 marginal 추가: 카테고리별 평균(All 열), specificity별 평균 + 전체 평균(All 행)
    recall_pivot["All"] = df.groupby("category")["hit"].mean()
    spec_mean = df.groupby("specificity")["hit"].mean().reindex(present_specs)
    spec_mean["All"] = df["hit"].mean()  # overall Recall@k
    recall_pivot.loc["All"] = spec_mean

    # 콘솔 출력
    print(f"\n=== Recall@{args.topk} by category × specificity ===")
    print(f"(source: {args.pred})")
    print(recall_pivot.round(4).to_string())
    print("\n=== Sample counts per cell ===")
    print(count_pivot.fillna(0).astype(int).to_string())
    print(f"\nOverall Recall@{args.topk}: {df['hit'].mean():.4f}  (n={len(df)})")

    # CSV 저장 (출력 파일명은 입력 추론 파일명에서 따온다)
    pred_stem = Path(args.pred).stem
    os.makedirs("exp/scores/devset", exist_ok=True)
    recall_path = f"exp/scores/devset/{pred_stem}_recall{args.topk}_heatmap.csv"
    count_path = f"exp/scores/devset/{pred_stem}_recall{args.topk}_counts.csv"
    recall_pivot.round(4).to_csv(recall_path)
    count_pivot.fillna(0).astype(int).to_csv(count_path)
    print(f"\nSaved: {recall_path}")
    print(f"Saved: {count_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-(category × specificity) Recall@k from a saved inference JSON."
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to inference JSON (exp/inference/devset/<tid>_<ts>.json)",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="exp/ground_truth/devset.json",
        help="Path to ground truth JSON",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
        help="HF dataset providing conversation_goal (category/specificity) labels",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--topk", type=int, default=20, help="Cutoff k for Recall@k (default 20)"
    )
    args = parser.parse_args()
    main(args)
