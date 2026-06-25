"""
Inference 결과를 읽기 좋은 로그로 변환.
모델 로딩 없음 — metadata DB만 사용.

Usage:
    python inspect_results.py --input exp/inference/devset/llama1b_hybrid_devset_0526_1124.json

    # 특정 턴만
    python inspect_results.py --input ... --turns 1 2

    # GT miss 케이스만
    python inspect_results.py --input ... --miss_only

    # 특정 세션만
    python inspect_results.py --input ... --session ba3da7b0

Output:
    exp/logs/inspect_{filename}_{timestamp}.log
"""

import os
import sys
import json
import math
import argparse
import datetime
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from mcrs.db_item import MusicCatalogDB

GT_PATH    = "exp/ground_truth/devset.json"
ITEM_DB    = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
SPLIT      = ["all_tracks"]
CORPUS     = ["track_name", "artist_name", "album_name", "release_date", "tag_list"]


def fmt_track(meta: dict, prefix: str = "    ") -> str:
    def lst(v):
        return v if isinstance(v, list) else [v]
    name   = ", ".join(lst(meta.get("track_name",  ["?"])))
    artist = ", ".join(lst(meta.get("artist_name", ["?"])))
    album  = ", ".join(lst(meta.get("album_name",  ["?"])))
    date   = str(meta.get("release_date", "?"))
    tags   = lst(meta.get("tag_list", []))[:10]
    pop    = meta.get("popularity", "?")
    return (
        f"{prefix}Name   : {name}\n"
        f"{prefix}Artist : {artist}\n"
        f"{prefix}Album  : {album}  ({date})\n"
        f"{prefix}Tags   : {', '.join(tags)}\n"
        f"{prefix}Pop    : {pop}"
    )


def main(args):
    print("Loading metadata DB...")
    item_db = MusicCatalogDB(ITEM_DB, SPLIT, CORPUS)

    print(f"Loading inference: {args.input}")
    predictions = json.load(open(args.input))
    gt_data     = json.load(open(GT_PATH))

    gt_map = {
        f"{r['session_id']}_{r['turn_number']}": r["ground_truth_track_id"]
        for r in gt_data
    }

    # Build user_query map
    print("Loading dataset for queries...")
    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    query_map = {}
    for item in db:
        df = pd.DataFrame(item["conversations"])
        for turn in range(1, 9):
            row = df[df["turn_number"] == turn]
            if not row.empty and row.iloc[0]["role"] == "user":
                query_map[f"{item['session_id']}_{turn}"] = row.iloc[0]["content"]

    turns_filter   = set(args.turns) if args.turns else None
    session_filter = args.session

    os.makedirs("exp/logs", exist_ok=True)
    ts       = datetime.datetime.now().strftime("%m%d_%H%M")
    basename = os.path.basename(args.input).replace(".json", "")
    log_path = f"exp/logs/inspect_{basename}_{ts}.log"

    stats = defaultdict(lambda: {"hit": 0, "miss": 0, "ranks": []})

    # Group by session for clean output
    from itertools import groupby
    predictions_sorted = sorted(predictions, key=lambda x: (x["session_id"], x["turn_number"]))

    with open(log_path, "w", encoding="utf-8") as f:

        def w(line=""):
            f.write(line + "\n")

        topk = len(predictions[0]["predicted_track_ids"])

        w("=" * 80)
        w(f"  File    : {args.input}")
        w(f"  TopK    : {topk}   Entries: {len(predictions)}")
        w(f"  Date    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        w("=" * 80)
        w()

        for session_id, group in groupby(predictions_sorted, key=lambda x: x["session_id"]):
            if session_filter and session_filter not in session_id:
                continue

            turns = list(group)
            user_id = turns[0]["user_id"]

            # Check if any turn in this session passes filter
            visible_turns = [
                t for t in turns
                if (turns_filter is None or t["turn_number"] in turns_filter)
            ]
            if not visible_turns:
                continue

            # Pre-check miss_only: skip session if no misses
            if args.miss_only:
                has_miss = False
                for t in visible_turns:
                    key   = f"{session_id}_{t['turn_number']}"
                    gt_id = gt_map.get(key)
                    if gt_id not in t["predicted_track_ids"]:
                        has_miss = True
                        break
                if not has_miss:
                    continue

            w("=" * 80)
            w(f"  SESSION  : {session_id}")
            w(f"  USER     : {user_id}")
            w("=" * 80)
            w()

            for t in visible_turns:
                turn   = t["turn_number"]
                key    = f"{session_id}_{turn}"
                gt_id  = gt_map.get(key)
                top_ids = t["predicted_track_ids"]

                gt_rank = next(
                    (i + 1 for i, tid in enumerate(top_ids) if tid == gt_id), None
                )

                if args.miss_only and gt_rank is not None:
                    continue

                user_query = query_map.get(key, "")

                w(f"  ┌─ TURN {turn} {'─' * 65}")
                w(f"  │")
                w(f"  │  QUERY : {user_query}")
                w(f"  │")

                # Ground truth block
                rank_str  = f"RANK #{gt_rank}" if gt_rank else f"MISS (not in top {topk})"
                gt_meta   = item_db.metadata_dict.get(gt_id, {}) if gt_id else {}
                w(f"  │  ── GROUND TRUTH [{rank_str}] ──────────────────────────────────")
                if gt_meta:
                    w(fmt_track(gt_meta, prefix="  │    "))
                else:
                    w(f"  │    track_id: {gt_id}  (not in metadata)")
                w(f"  │")

                # Predicted tracks
                w(f"  │  ── TOP {topk} PREDICTIONS ──────────────────────────────────────")
                for rank, tid in enumerate(top_ids, start=1):
                    meta   = item_db.metadata_dict.get(tid, {})
                    def lst(v):
                        return v if isinstance(v, list) else [v]
                    name   = ", ".join(lst(meta.get("track_name",  ["?"])))
                    artist = ", ".join(lst(meta.get("artist_name", ["?"])))
                    tags   = ", ".join(lst(meta.get("tag_list",    []))[:4])
                    pop    = meta.get("popularity", "?")
                    marker = "  ◀◀◀ GROUND TRUTH" if tid == gt_id else ""
                    w(f"  │  #{rank:>2}  {name[:38]:<38}  {artist[:22]:<22}  pop={str(pop):<5}  [{tags}]{marker}")

                w(f"  │")
                w(f"  └{'─' * 72}")
                w()

                # Stats
                if gt_rank:
                    stats[turn]["hit"] += 1
                    stats[turn]["ranks"].append(gt_rank)
                else:
                    stats[turn]["miss"] += 1

        # Summary
        w()
        w("=" * 80)
        w("  SUMMARY")
        w("=" * 80)
        w(f"  {'Turn':<6} {'Hit@1':<8} {'Hit@10':<9} {'Hit@20':<9} {f'Hit@{topk}':<9} {'nDCG@20':<10} {'AvgRank'}")
        w("  " + "─" * 68)
        all_ranks, all_n = [], 0
        for turn in sorted(stats.keys()):
            s  = stats[turn]
            rs = s["ranks"]
            n  = s["hit"] + s["miss"]
            all_n    += n
            all_ranks += rs
            h1   = sum(1 for r in rs if r <= 1)
            h10  = sum(1 for r in rs if r <= 10)
            h20  = sum(1 for r in rs if r <= 20)
            htk  = sum(1 for r in rs if r <= topk)
            ndcg = sum(1/math.log2(r+1) for r in rs if r <= 20) / n if n else 0
            avg  = sum(rs) / len(rs) if rs else 0
            w(f"  {turn:<6} {h1/n*100:<7.1f}% {h10/n*100:<8.1f}% "
              f"{h20/n*100:<8.1f}% {htk/n*100:<8.1f}% {ndcg:<10.4f} {avg:.1f}")
        w("  " + "─" * 68)
        if all_n:
            h1   = sum(1 for r in all_ranks if r <= 1)
            h20  = sum(1 for r in all_ranks if r <= 20)
            htk  = sum(1 for r in all_ranks if r <= topk)
            ndcg = sum(1/math.log2(r+1) for r in all_ranks if r <= 20) / all_n
            w(f"  {'ALL':<6} {h1/all_n*100:<7.1f}% {'':8} "
              f"{h20/all_n*100:<8.1f}% {htk/all_n*100:<8.1f}% {ndcg:<10.4f}")
        w()
        w(f"  Log → {log_path}")

    print(f"\nDone. → {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True,
                        help="inference JSON 경로")
    parser.add_argument("--turns",    type=int, nargs="+", default=None,
                        help="특정 턴만 출력 (예: --turns 1 2 3)")
    parser.add_argument("--miss_only", action="store_true",
                        help="GT가 predicted list 밖인 케이스만")
    parser.add_argument("--session",  default=None,
                        help="특정 session UUID 포함 케이스만 (부분 매칭)")
    args = parser.parse_args()
    main(args)
