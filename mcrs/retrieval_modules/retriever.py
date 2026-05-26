import os
import json
import hashlib
from typing import List, Optional
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import ClapModel, ClapProcessor, AutoTokenizer, AutoModelForCausalLM
from .extract_query_comp import extract_structured_query


class CLAP_MODEL:
    """Text-to-audio retriever using pre-computed track audio embeddings.

    Loads the CLAP text encoder from HuggingFace and aligns it against
    pre-computed audio embeddings stored in `talkpl-ai/TalkPlayData-2-Track-Embeddings`.
    Audio embeddings are cached locally to avoid re-downloading.
    """

    def __init__(self, model_name: str = "laion/larger_clap_music",
        embedding_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split_types: list[str] = ["all_tracks"],
        cache_dir: str = "./cache", device: str | None = None,) -> None:
        self.model_name = model_name
        self.embedding_dataset = embedding_dataset
        self.split_types = split_types
        self.cache_dir = cache_dir
        self.index_dir = os.path.join(cache_dir, "clap")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = ClapModel.from_pretrained(model_name)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model.to(self.device).eval()

        self.embeddings, self.track_ids = self._load_or_build_index()
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}

    def _load_or_build_index(self):
        emb_path = os.path.join(self.index_dir, "embeddings.pt")
        ids_path = os.path.join(self.index_dir, "track_ids.json")

        if os.path.exists(emb_path) and os.path.exists(ids_path):
            embeddings = torch.load(emb_path, map_location="cpu")
            with open(ids_path) as f:
                track_ids = json.load(f)
            return embeddings, track_ids

        dataset = load_dataset(self.embedding_dataset)
        splits = [dataset[s] for s in self.split_types if s in dataset]
        full_dataset = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

        track_ids_raw = list(full_dataset["track_id"])
        raw_embeddings = list(full_dataset["audio-laion_clap"])  # list of 512-dim float lists

        # Filter out entries with missing/empty embeddings
        valid = [(tid, emb) for tid, emb in zip(track_ids_raw, raw_embeddings) if emb and len(emb) == 512]
        track_ids, raw_embeddings = zip(*valid)
        track_ids = list(track_ids)
        print(f"CLAP: loaded {len(track_ids)} tracks from split(s) {self.split_types}")
        embeddings = torch.tensor(raw_embeddings, dtype=torch.float32)
        embeddings = F.normalize(embeddings, p=2, dim=1) # normalize for cosine similarity

        os.makedirs(self.index_dir, exist_ok=True)
        torch.save(embeddings, emb_path)
        with open(ids_path, "w") as f:
            json.dump(track_ids, f)

        return embeddings, track_ids

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a text query into a normalized CLAP text embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_inputs = {k: v.to(self.device) for k, v in inputs.items()
                       if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            text_outputs = self.model.text_model(**text_inputs)
            pooled = text_outputs.pooler_output          # [1, hidden_dim]
            text_emb = self.model.text_projection(pooled)  # [1, 512] This maps to the same space as audio embeddings
            text_emb = F.normalize(text_emb, p=2, dim=1)
        return text_emb.squeeze(0).cpu()  # [512]

    def text_to_audio_retrieval(self, mood_text: str, topk: int = 20) -> List[str]:
        """Retrieve top-k track IDs by cosine similarity between text and audio embeddings."""
        query_emb = self.encode_text(mood_text)
        scores = torch.matmul(self.embeddings, query_emb)  # [N]
        topk = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids[i] for i in top_indices]


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


class HYBRID_MODEL:
    """Hybrid retriever combining BGE text scores with CLAP audio scores.

    BGE retrieves based on metadata text similarity; CLAP retrieves based on
    audio-semantic similarity from keywords extracted by Qwen3-4B.
    Final score = alpha * normalize(bge_score) + beta * normalize(clap_score).

    The interface matches BGE_MODEL so it is a drop-in for `crs_baseline.py`.
    """

    def __init__(self, bge_model, clap_model: CLAP_MODEL,
        alpha: float = 0.7, beta: float = 0.3,
        keyword_model_name: str = "Qwen/Qwen3-4B",
        keyword_cache_path: Optional[str] = None,
        reranker: Optional[RERANKER] = None,
    ) -> None:
        self.bge_model = bge_model
        self.clap_model = clap_model
        self.alpha = alpha
        self.beta = beta
        self.track_ids = bge_model.track_ids  # authoritative track order

        # Load precomputed keyword cache if provided; otherwise load Qwen at runtime.
        self.keyword_cache: dict = {}
        self.keyword_model = None
        self.keyword_tokenizer = None
        self.keyword_device = "cuda" if torch.cuda.is_available() else "cpu"

        if keyword_cache_path and os.path.exists(keyword_cache_path):
            with open(keyword_cache_path, "r", encoding="utf-8") as f:
                self.keyword_cache = json.load(f)
            print(f"[HYBRID] Loaded keyword cache: {len(self.keyword_cache)} entries from {keyword_cache_path}")
        else:
            print(f"[HYBRID] No keyword cache found — loading Qwen ({keyword_model_name})")
            self.keyword_tokenizer, self.keyword_model = self._load_keyword_model(keyword_model_name)

        # Pre-align CLAP embeddings to BGE's track order.
        # Tracks missing from CLAP get a zero vector (CLAP contributes nothing).
        clap_dim = clap_model.embeddings.shape[1]
        aligned = torch.zeros(len(self.track_ids), clap_dim)
        for i, tid in enumerate(self.track_ids):
            if tid in clap_model.track_id_to_idx:
                j = clap_model.track_id_to_idx[tid]
                aligned[i] = clap_model.embeddings[j]
        self.clap_aligned = aligned  # [N_bge, clap_dim]
        self.reranker = reranker

    def _load_keyword_model(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        model.to(self.keyword_device).eval()
        return tokenizer, model

    def _lookup_or_extract(self, query: str) -> dict:
        """Return precomputed components if cached, otherwise run Qwen (or fallback)."""
        key = hashlib.md5(query.encode("utf-8")).hexdigest()
        if key in self.keyword_cache:
            return self.keyword_cache[key]
        if self.keyword_model is not None:
            lm_components = (self.keyword_model, self.keyword_tokenizer, self.keyword_device)
            return extract_structured_query(query, lm_components)
        # Cache miss with no Qwen loaded: use last user message as raw fallback
        from .extract_query_comp import _last_user_message
        fallback = _last_user_message(query)
        print(f"[HYBRID] Cache miss — using raw fallback: {fallback[:80]}")
        return {"direct_request": None, "bge_query": fallback, "clap_keywords": fallback, "rejected": []}

    def _bge_scores(self, query: str) -> torch.Tensor:
        """Compute cosine similarity scores between query and all BGE track embeddings."""
        instruction_query = self.bge_model.QUERY_INSTRUCTION + query
        self.bge_model.model.eval()
        with torch.no_grad():
            batch = self.bge_model.tokenizer(
                [instruction_query],
                padding=True,
                truncation=True,
                max_length=self.bge_model.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.bge_model.device) for k, v in batch.items()}
            outputs = self.bge_model.model(**batch)
            query_emb = outputs.last_hidden_state[:, 0]
            query_emb = F.normalize(query_emb, p=2, dim=1).cpu().squeeze(0)
        return torch.matmul(self.bge_model.embeddings, query_emb)  # [N]

    def _clap_scores(self, mood_text: str) -> torch.Tensor:
        """Compute cosine similarity scores between keywords and all CLAP audio embeddings."""
        query_emb = self.clap_model.encode_text(mood_text)  # [dim]
        return torch.matmul(self.clap_aligned, query_emb)   # [N]

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-9:
            return torch.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def _filter_rejected(self, candidates: List[str], rejected: List[str]) -> List[str]:
        """Remove tracks whose artist appears in the rejected list."""
        if not rejected:
            return candidates
        rejected_lower = [r.lower() for r in rejected]
        filtered = []
        for tid in candidates:
            meta = self.bge_model.metadata_dict.get(tid, {})
            artists = meta.get("artist_name", [])
            artists = [artists] if isinstance(artists, str) else artists
            if not any(rej in a.lower() for rej in rejected_lower for a in artists):
                filtered.append(tid)
        return filtered if filtered else candidates  # fallback: don't return empty

    def text_to_item_retrieval(self, query: str, topk: int, user_id=None) -> List[str]:
        """Retrieve top-k tracks, optionally re-ranked by user CF-BPR affinity."""
        components = self._lookup_or_extract(query)
        bge_scores = self._normalize(self._bge_scores(components["bge_query"]))
        # clap_scores = self._normalize(self._clap_scores(components["clap_keywords"]))
        # combined = self.alpha * bge_scores + self.beta * clap_scores
        combined = bge_scores

        # Pull 100 candidates for re-ranking (topk*5 if no reranker)
        n_candidates = min(100 if self.reranker else topk * 5, combined.shape[0])
        top_result = torch.topk(combined, k=n_candidates)
        top_indices = top_result.indices.tolist()
        top_scores = top_result.values  # keep scores for blended re-ranking

        candidates = [self.track_ids[i] for i in top_indices]
        candidates = self._filter_rejected(candidates, components["rejected"])

        if self.reranker is not None:
            # Align scores to filtered candidate order
            score_map = {self.track_ids[i]: top_scores[pos] for pos, i in enumerate(top_indices)}
            candidate_scores = torch.tensor([score_map[tid].item() for tid in candidates])
            return self.reranker.rerank(candidates, candidate_scores, user_id, topk)
        return candidates[:topk]

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int, user_ids: List = None) -> List[List[str]]:
        """Retrieve top-k tracks for multiple queries."""
        if user_ids is None:
            user_ids = [None] * len(queries)
        return [self.text_to_item_retrieval(q, topk, uid) for q, uid in zip(queries, user_ids)]
