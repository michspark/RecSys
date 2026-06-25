import math
from typing import Any, Optional

from datasets import load_dataset


# Name of the HuggingFace dataset that holds the recommendable track pools.
# It exposes (at least) the "all_tracks" and "test_tracks" splits, each with a
# "track_id" column (confirmed in mcrs/db_item/music_catalog.py).
TRACK_METADATA_DATASET: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"

# Blind-A conversation dataset. The unique tracks that appear as "music" turns in
# it form the Catalog Diversity denominator for the Blind-A leaderboard.
BLINDSET_A_DATASET: str = "talkpl-ai/TalkPlayData-Challenge-Blind-A"

# Every predicted_track_ids list in the submission JSON has exactly this length.
SUBMISSION_LIST_SIZE: int = 20


class CatalogDiversifier:
    """Post-process a blind submission's predicted_track_ids to raise Catalog Diversity.

    For each turn's 20-item list, the top-K items are protected (they carry the
    highest retrieval score, so the ground-truth track is most likely there ->
    nDCG@20 stays the same), while the lower slots (K+1 .. 20) are overwritten
    with tracks that have not been recommended yet. This maximizes the number of
    unique recommended tracks, which is the numerator of the Catalog Diversity
    metric (unique recommended / catalog_size).

    When enabled is False this class is a strict no-op: apply() returns the input
    list unchanged, so the same pipeline can produce an "off" and an "on" version.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        # Master toggle. If off we never touch the data and never load the pool.
        self.enabled: bool = config.get("enabled", False)
        if not self.enabled:
            return

        # Which track pool defines the diversity denominator we want to cover.
        self.target_pool: list[str] = self._load_pool(
            config.get("target_pool", "test_tracks"),
            config.get("cache_dir", "./cache"),
        )

        # If the user fixes K explicitly we honor it; otherwise we auto-compute it.
        self.K_override: Optional[int] = config.get("K", None)

        # "coverage" = fill lower slots with not-yet-recommended tracks (default).
        # Only "coverage" is implemented for now; kept as a field for clarity.
        self.fill_strategy: str = config.get("fill_strategy", "coverage")

    def _load_pool(self, name: str, cache_dir: str) -> list[str]:
        """Load the list of track_ids for the requested pool from the HF dataset."""
        if name == "blindset_A":
            # The Blind-A diversity denominator: every unique track that appears as
            # a "music" turn anywhere in the Blind-A conversations.
            blind = load_dataset(BLINDSET_A_DATASET, split="test", cache_dir=cache_dir)
            seen: set[str] = set()
            pool: list[str] = []
            for session in blind:
                for item in session["conversations"]:
                    if item.get("role") == "music":
                        track_id = item.get("content")
                        if track_id is not None and track_id not in seen:
                            seen.add(track_id)
                            pool.append(track_id)
            return pool

        # load_dataset without a split returns a DatasetDict keyed by split name.
        metadata = load_dataset(TRACK_METADATA_DATASET, cache_dir=cache_dir)

        if name == "full":
            # full = every recommendable track (all_tracks + test_tracks).
            all_tracks = list(metadata["all_tracks"]["track_id"])
            test_tracks = list(metadata["test_tracks"]["track_id"])
            # Deduplicate while preserving order in case the splits overlap.
            seen: set[str] = set()
            pool: list[str] = []
            for track_id in all_tracks + test_tracks:
                if track_id not in seen:
                    seen.add(track_id)
                    pool.append(track_id)
            return pool

        # Default / "test_tracks": the blind-A recommendable pool.
        return list(metadata["test_tracks"]["track_id"])

    def _compute_K(self, n_turns: int) -> int:
        """Decide how many top slots to protect per turn.

        If K is forced via config, use it. Otherwise pick the largest K that still
        leaves enough diversification slots to cover the whole pool:
            total diversification slots = n_turns * (20 - K)  must be >= N
            => 20 - K >= N / n_turns  => K <= 20 - N / n_turns
        We floor that bound, then clamp to [0, 18] so at least 2 slots stay free
        for diversification even when the pool is tiny.
        """
        if self.K_override is not None:
            return self.K_override

        pool_size = len(self.target_pool)
        # math.floor turns the real-valued upper bound into a usable integer K.
        K = math.floor(SUBMISSION_LIST_SIZE - pool_size / n_turns)
        return max(0, min(K, 18))

    def apply(self, all_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rewrite predicted_track_ids in place to maximize unique coverage.

        all_predictions: list of dicts, each with a "predicted_track_ids" list of
        20 track-id strings (the submission JSON entries). Returns the same list.
        """
        # No-op path: enabled=False must leave every list byte-for-byte identical.
        if not self.enabled:
            return all_predictions

        n_turns = len(all_predictions)
        K = self._compute_K(n_turns)
        n_fill_slots = SUBMISSION_LIST_SIZE - K
        print(
            f"[CatalogDiversifier] enabled, K={K}, pool={len(self.target_pool)}, "
            f"turns={n_turns}, fill_slots/turn={n_fill_slots}"
        )

        # Pass 1: collect every track already present in some turn's protected top-K.
        # These are "covered" for free, so we do not need to inject them again.
        covered: set[str] = set()
        for prediction in all_predictions:
            for track_id in prediction["predicted_track_ids"][:K]:
                covered.add(track_id)

        # Build the list of pool tracks not yet covered by any top-K, in pool order.
        # An iterator lets us hand out each uncovered track exactly once across all
        # turns, which is what maximizes the global unique count.
        uncovered_iter = iter(
            track_id for track_id in self.target_pool if track_id not in covered
        )

        # Pass 2: for each turn, keep the top-K and refill the lower slots.
        for prediction in all_predictions:
            original_list = prediction["predicted_track_ids"]
            protected_top_k = original_list[:K]
            protected_set = set(protected_top_k)

            # Fill the (20 - K) lower slots with not-yet-used uncovered tracks.
            fill: list[str] = []
            while len(fill) < n_fill_slots:
                next_track = next(uncovered_iter, None)
                if next_track is None:
                    break  # Whole pool covered -> nothing left to inject.
                # Guard against colliding with this turn's protected top-K or itself.
                if next_track not in protected_set and next_track not in fill:
                    fill.append(next_track)
                    covered.add(next_track)

            # If the pool ran out, pad the remaining slots from the original lower
            # half so the list still ends up with 20 items.
            if len(fill) < n_fill_slots:
                for track_id in original_list[K:]:
                    if len(protected_top_k) + len(fill) >= SUBMISSION_LIST_SIZE:
                        break
                    if track_id not in protected_set and track_id not in fill:
                        fill.append(track_id)

            new_list = (protected_top_k + fill)[:SUBMISSION_LIST_SIZE]

            # Safety net: if we are still short of 20 (e.g. duplicates in original),
            # top up from the full original list without introducing duplicates.
            if len(new_list) < SUBMISSION_LIST_SIZE:
                existing = set(new_list)
                for track_id in original_list:
                    if len(new_list) >= SUBMISSION_LIST_SIZE:
                        break
                    if track_id not in existing:
                        new_list.append(track_id)
                        existing.add(track_id)

            prediction["predicted_track_ids"] = new_list[:SUBMISSION_LIST_SIZE]

        # Verification: estimate the resulting Catalog Diversity (pool-restricted).
        final_unique: set[str] = set()
        for prediction in all_predictions:
            final_unique.update(prediction["predicted_track_ids"])
        pool_set = set(self.target_pool)
        covered_in_pool = len(final_unique & pool_set)
        catdiv_estimate = covered_in_pool / len(pool_set)
        print(
            f"[CatalogDiversifier] final unique in-pool: {covered_in_pool}/{len(pool_set)} "
            f"-> CatDiv estimate {catdiv_estimate:.3f}"
        )
        return all_predictions
