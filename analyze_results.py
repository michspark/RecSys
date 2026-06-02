"""
Detailed analysis of inference results vs ground truth.

Usage:
    # Single file
    python analyze_results.py --pred exp/inference/devset/llama1b_hybrid_devset_0526_0213.json

    # Compare two files side-by-side
    python analyze_results.py \
        --pred exp/inference/devset/llama1b_hybrid_devset_0526_0213.json \
        --pred2 exp/inference/devset/llama1b_hybrid_devset_0525_2028.json

    # Show per-session details (misses only)
    python analyze_results.py --pred ... --show_misses
"""

import json
import argparse
import math
from collections import defaultdict


GT_PATH = "exp/ground_truth/devset.json"


# ─── helpers ────────────────────────────────────────────────────────────────

def load_pred(path: str) -> dict:
    """Load inference JSON → {session_id_turn: predicted_track_ids}"""
    data = json.load(open(path))
    return {f"{r['session_id']}_{r['turn_number']}": r["predicted_track_ids"] for r in data}


def load_gt(path: str) -> dict:
    """Load ground truth JSON → {session_id_turn: ground_truth_track_id}"""
    data = json.load(open(path))
    return {f"{r['session_id']}_{r['turn_number']}": r["ground_truth_track_id"] for r in data}


def hit_rank(gt_id: str, preds: list) -> int:
    """Return 1-based rank of gt_id in preds, or 0 if not found."""
    try:
        return preds.index(gt_id) + 1
    except ValueError:
        return 0


def ndcg_at_k(rank: int, k: int) -> float:
    if rank == 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


# ─── per-turn stats ──────────────────────────────────────────────────────────

def compute_turn_stats(pred_map: dict, gt_map: dict):
    """Returns per-turn breakdown dict."""
    stats = defaultdict(lambda: {"hits": [], "ranks": [], "misses": 0})

    for key, gt_id in gt_map.items():
        turn = int(key.split("_")[-1])
        preds = pred_map.get(key, [])
        rank = hit_rank(gt_id, preds)
        if rank > 0:
            stats[turn]["hits"].append(rank)
            stats[turn]["ranks"].append(rank)
        else:
            stats[turn]["misses"] += 1

    return stats


def print_turn_table(stats: dict, label: str = ""):
    n_turns = sorted(stats.keys())
    total_hits = sum(len(stats[t]["hits"]) for t in n_turns)
    total_all  = sum(len(stats[t]["hits"]) + stats[t]["misses"] for t in n_turns)

    if label:
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")

    header = f"{'Turn':<6} {'Hit@1':<8} {'Hit@5':<8} {'Hit@10':<9} {'Hit@20':<9} {'nDCG@20':<10} {'AvgRank':<9} {'Total'}"
    print(header)
    print("─" * len(header))

    all_ndcg, all_hit1, all_hit5, all_hit10, all_hit20 = [], [], [], [], []

    for t in n_turns:
        ranks = stats[t]["ranks"]
        n_miss = stats[t]["misses"]
        n_total = len(ranks) + n_miss

        h1  = sum(1 for r in ranks if r <= 1)
        h5  = sum(1 for r in ranks if r <= 5)
        h10 = sum(1 for r in ranks if r <= 10)
        h20 = sum(1 for r in ranks if r <= 20)
        ndcg = sum(ndcg_at_k(r, 20) for r in ranks) / n_total if n_total else 0
        avg_rank = sum(ranks) / len(ranks) if ranks else 0

        all_hit1.extend([1 if r <= 1 else 0 for r in ranks] + [0]*n_miss)
        all_hit5.extend([1 if r <= 5 else 0 for r in ranks] + [0]*n_miss)
        all_hit10.extend([1 if r <= 10 else 0 for r in ranks] + [0]*n_miss)
        all_hit20.extend([1 if r <= 20 else 0 for r in ranks] + [0]*n_miss)
        all_ndcg.extend([ndcg_at_k(r, 20) for r in ranks] + [0]*n_miss)

        print(f"{t:<6} {h1/n_total*100:<7.1f}% {h5/n_total*100:<7.1f}% "
              f"{h10/n_total*100:<8.1f}% {h20/n_total*100:<8.1f}% "
              f"{ndcg:<9.4f}  {avg_rank:<8.1f} {n_total}")

    print("─" * len(header))
    n = len(all_ndcg)
    print(f"{'ALL':<6} {sum(all_hit1)/n*100:<7.1f}% {sum(all_hit5)/n*100:<7.1f}% "
          f"{sum(all_hit10)/n*100:<8.1f}% {sum(all_hit20)/n*100:<8.1f}% "
          f"{sum(all_ndcg)/n:<9.4f}  {'—':<8} {n}")


# ─── rank distribution ───────────────────────────────────────────────────────

def print_rank_distribution(stats: dict):
    all_ranks = [r for t in stats for r in stats[t]["ranks"]]
    buckets = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 15), (16, 20)]
    n_miss = sum(stats[t]["misses"] for t in stats)
    total = len(all_ranks) + n_miss

    print("\n  Rank distribution (of hits):")
    for lo, hi in buckets:
        cnt = sum(1 for r in all_ranks if lo <= r <= hi)
        bar = "█" * int(cnt / total * 50)
        label = f"#{lo}" if lo == hi else f"#{lo}-{hi}"
        print(f"  {label:<8} {cnt:>5}  {cnt/total*100:>5.1f}%  {bar}")
    print(f"  {'miss':<8} {n_miss:>5}  {n_miss/total*100:>5.1f}%")


# ─── comparison ─────────────────────────────────────────────────────────────

def compare_two(pred1: dict, pred2: dict, gt_map: dict, label1: str, label2: str):
    """Show keys where predictions differ in rank outcome."""
    improved, degraded, same = 0, 0, 0
    examples_improved, examples_degraded = [], []

    for key, gt_id in gt_map.items():
        r1 = hit_rank(gt_id, pred1.get(key, []))
        r2 = hit_rank(gt_id, pred2.get(key, []))
        turn = int(key.split("_")[-1])

        if r1 == r2:
            same += 1
        elif (r2 > 0 and (r1 == 0 or r2 < r1)):
            improved += 1
            if len(examples_improved) < 5:
                examples_improved.append((key, r1, r2, turn))
        else:
            degraded += 1
            if len(examples_degraded) < 5:
                examples_degraded.append((key, r1, r2, turn))

    total = improved + degraded + same
    print(f"\n  {label2} vs {label1}:")
    print(f"  Improved: {improved:>5} ({improved/total*100:.1f}%)")
    print(f"  Degraded: {degraded:>5} ({degraded/total*100:.1f}%)")
    print(f"  Same:     {same:>5} ({same/total*100:.1f}%)")

    if examples_improved:
        print(f"\n  Sample improvements (session_turn | old rank → new rank):")
        for key, r1, r2, t in examples_improved:
            old = f"#{r1}" if r1 > 0 else "miss"
            new = f"#{r2}" if r2 > 0 else "miss"
            print(f"    turn={t}  {old} → {new}  [{key[:24]}…]")

    if examples_degraded:
        print(f"\n  Sample degradations:")
        for key, r1, r2, t in examples_degraded:
            old = f"#{r1}" if r1 > 0 else "miss"
            new = f"#{r2}" if r2 > 0 else "miss"
            print(f"    turn={t}  {old} → {new}  [{key[:24]}…]")


# ─── miss analysis ───────────────────────────────────────────────────────────

def print_misses(pred_map: dict, gt_map: dict, n: int = 10):
    """Print first n sessions where turn 1 is a miss."""
    print(f"\n  First {n} Turn-1 misses (gt not in top-20):")
    shown = 0
    for key, gt_id in gt_map.items():
        if not key.endswith("_1"):
            continue
        preds = pred_map.get(key, [])
        if gt_id not in preds:
            session = key[:-2]
            print(f"    session={session[:36]}  gt={gt_id[:8]}…  pred[0]={preds[0][:8] if preds else '—'}…")
            shown += 1
            if shown >= n:
                break


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred",  required=True, help="Primary inference JSON")
    parser.add_argument("--pred2", default=None,  help="Second inference JSON for comparison")
    parser.add_argument("--gt",    default=GT_PATH)
    parser.add_argument("--show_misses", action="store_true")
    args = parser.parse_args()

    gt_map   = load_gt(args.gt)
    pred_map = load_pred(args.pred)
    label1   = args.pred.split("/")[-1].replace(".json", "")

    stats = compute_turn_stats(pred_map, gt_map)
    print_turn_table(stats, label=label1)
    print_rank_distribution(stats)

    if args.pred2:
        pred_map2 = load_pred(args.pred2)
        label2    = args.pred2.split("/")[-1].replace(".json", "")
        stats2    = compute_turn_stats(pred_map2, gt_map)
        print_turn_table(stats2, label=label2)
        compare_two(pred_map, pred_map2, gt_map, label1, label2)

    if args.show_misses:
        print_misses(pred_map, gt_map)


if __name__ == "__main__":
    main()
