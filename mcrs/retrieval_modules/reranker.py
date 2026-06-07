import torch
import torch.nn.functional as F
import os
import json
from typing import List
from datasets import load_dataset, concatenate_datasets


class RERANKER:
    """Re-ranks candidates using user CF-BPR ↔ track CF-BPR cosine similarity.

    Both embeddings live in the same 128-dim collaborative-filtering space so
    cosine similarity directly reflects listening-history affinity.
    """

    def __init__(self,
        track_embedding_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        user_embedding_dataset: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        split_types: list = ["all_tracks"],
        cache_dir: str = "./precomputed/reranker",
    ) -> None:
        self.index_dir = cache_dir
        self.track_embeddings, self.track_ids = self._load_track_embeddings(track_embedding_dataset, split_types)
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}
        self.user_embeddings = self._load_user_embeddings(user_embedding_dataset)

    def _load_track_embeddings(self, dataset_name: str, split_types: list):
        emb_path = os.path.join(self.index_dir, "cf_bpr_track.pt")
        ids_path = os.path.join(self.index_dir, "cf_bpr_track_ids.json")
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            embeddings = torch.load(emb_path, map_location="cpu")
            with open(ids_path) as f:
                track_ids = json.load(f)
            print(f"[ReRanker] Loaded {len(track_ids)} track CF-BPR embeddings from cache")
            return embeddings, track_ids

        dataset = load_dataset(dataset_name)
        splits = [dataset[s] for s in split_types if s in dataset]
        full = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

        raw_ids = list(full["track_id"])
        raw_embs = list(full["cf-bpr"])
        valid = [(tid, emb) for tid, emb in zip(raw_ids, raw_embs) if emb and len(emb) == 128]
        track_ids, raw_embs = zip(*valid)
        track_ids = list(track_ids)
        embeddings = F.normalize(torch.tensor(raw_embs, dtype=torch.float32), p=2, dim=1)

        os.makedirs(self.index_dir, exist_ok=True)
        torch.save(embeddings, emb_path)
        with open(ids_path, "w") as f:
            json.dump(track_ids, f)
        print(f"[ReRanker] Built track CF-BPR index: {len(track_ids)} tracks")
        return embeddings, track_ids

    def _load_user_embeddings(self, dataset_name: str) -> dict:
        cache_path = os.path.join(self.index_dir, "cf_bpr_users.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                raw = json.load(f)
            user_embs = {uid: torch.tensor(v, dtype=torch.float32) for uid, v in raw.items()}
            print(f"[ReRanker] Loaded {len(user_embs)} user CF-BPR embeddings from cache")
            return user_embs

        dataset = load_dataset(dataset_name)
        user_embs = {}
        for split in dataset.values():
            for row in split:
                uid = str(row["user_id"])
                emb = row["cf-bpr"]
                if emb and len(emb) == 128:
                    user_embs[uid] = F.normalize(
                        torch.tensor(emb, dtype=torch.float32).unsqueeze(0), p=2, dim=1
                    ).squeeze(0)

        os.makedirs(self.index_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({uid: v.tolist() for uid, v in user_embs.items()}, f)
        print(f"[ReRanker] Loaded {len(user_embs)} user CF-BPR embeddings")
        return user_embs

    _log_count = 0  # class-level counter to limit log verbosity

    def rerank(self, candidate_ids: List[str], retrieval_scores: torch.Tensor,
               user_id, topk: int, cf_weight: float = 0.3) -> List[str]:
        """Re-rank by blending retrieval score with CF-BPR user affinity.

        final_score = (1 - cf_weight) * retrieval_score + cf_weight * cf_bpr_score
        Both scores are min-max normalized before blending.
        cf_weight=0 → pure retrieval order; cf_weight=1 → pure CF-BPR order.
        """
        uid = str(user_id) if user_id is not None else None
        if uid is None or uid not in self.user_embeddings:
            if RERANKER._log_count < 3:
                print(f"[ReRanker] user_id '{uid}' not in embeddings — skipping rerank")
                RERANKER._log_count += 1
            return candidate_ids[:topk]

        user_emb = self.user_embeddings[uid]  # [128]
        valid_mask = [tid in self.track_id_to_idx for tid in candidate_ids]
        valid = [(tid, self.track_id_to_idx[tid], score)
                 for tid, score, ok in zip(candidate_ids, retrieval_scores.tolist(), valid_mask) if ok]
        no_emb = [tid for tid, ok in zip(candidate_ids, valid_mask) if not ok]

        if not valid:
            return candidate_ids[:topk]

        tids, indices, ret_scores = zip(*valid)
        track_embs = self.track_embeddings[list(indices)]   # [N, 128]
        cf_scores = torch.matmul(track_embs, user_emb)     # [N]

        # Min-max normalize both score vectors
        def _minmax(t):
            mn, mx = t.min(), t.max()
            return (t - mn) / (mx - mn + 1e-9)

        ret_norm = _minmax(torch.tensor(ret_scores))
        cf_norm  = _minmax(cf_scores)
        combined = (1 - cf_weight) * ret_norm + cf_weight * cf_norm

        order = torch.argsort(combined, descending=True).tolist()
        reranked = [tids[i] for i in order] + no_emb

        if RERANKER._log_count < 3:
            moved = sum(1 for b, a in zip(candidate_ids[:topk], reranked[:topk]) if b != a)
            print(f"[ReRanker] user={uid[:8]}… | candidates={len(candidate_ids)} | "
                  f"positions changed: {moved}/{topk} | cf_weight={cf_weight}")
            RERANKER._log_count += 1

        return reranked[:topk]
