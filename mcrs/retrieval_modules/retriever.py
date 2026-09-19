import os
import re
import json
import hashlib
from typing import List, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from .extract_query_comp import extract_structured_query
from .clap import CLAP_MODEL
from .reranker import RERANKER

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


class ANCHOR_CF_MODEL:
    """BGE cumulative query + anchor (metadata vector blend) + cf-bpr (score fusion).

    - query: the cumulative conversation (retrieval_input) encoded with BGE (with QUERY_INSTRUCTION)
    - anchor: metadata of the previous (positive) recommended track, encoded with BGE, blended in
      with weight alpha and re-normalized
    - cf-bpr: mean cf-bpr vector of the accumulated (positive) tracks vs. every track's cf-bpr,
      fused at score level via z-score with weight beta
    anchor/positive/exclude/turn are injected by the pipeline (batch_chat); without them it runs query-only.
    """

    def __init__(self, bge_model, cf_cache_dir: str = "./precomputed/reranker",
                 beta: float = 0.2, alpha_start: float = 0.25, alpha_step: float = 0.05,
                 alpha_cap: float = 0.60) -> None:
        self.bge = bge_model
        self.device = bge_model.device
        self.track_ids = bge_model.track_ids
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}
        self.track_embs = bge_model.embeddings.to(self.device)   # [N, 1024], L2-normalized
        self.beta = beta
        self.alpha_start, self.alpha_step, self.alpha_cap = alpha_start, alpha_step, alpha_cap
        self.accepts_anchor = True   # marker telling batch_chat to pass anchor info

        # cf-bpr track index (reuses the normalized cache built by RERANKER)
        cf_embs = torch.load(os.path.join(cf_cache_dir, "cf_bpr_track.pt"), map_location="cpu")
        with open(os.path.join(cf_cache_dir, "cf_bpr_track_ids.json")) as f:
            cf_track_ids = json.load(f)
        self.cf_embs = cf_embs.to(self.device)                   # [M, 128], L2-normalized
        self._cf_id_to_idx = {tid: i for i, tid in enumerate(cf_track_ids)}
        # BGE track order -> cf index (-1 if the track has no cf embedding)
        cf_pos = [self._cf_id_to_idx.get(tid, -1) for tid in self.track_ids]
        self.cf_pos = torch.tensor(cf_pos, dtype=torch.long, device=self.device)
        self.valid_cf = self.cf_pos >= 0
        print(f"[ANCHOR_CF] BGE tracks={len(self.track_ids)} | cf-bpr aligned={int(self.valid_cf.sum())} "
              f"| beta={beta} alpha={alpha_start}->{alpha_cap}")

    @torch.no_grad()
    def _encode(self, text: str, is_query: bool) -> torch.Tensor:
        """BGE CLS encoding + L2 normalization. Queries get the instruction prefix."""
        if is_query:
            text = self.bge.QUERY_INSTRUCTION + text
        batch = self.bge.tokenizer([text], padding=True, truncation=True,
                                   max_length=self.bge.max_length, return_tensors="pt").to(self.bge.device)
        emb = self.bge.model(**batch).last_hidden_state[:, 0]
        return F.normalize(emb, p=2, dim=1).squeeze(0).to(self.device)

    def _encode_anchor(self, track_id: str) -> Optional[torch.Tensor]:
        """Encode anchor track metadata in the same format as the candidate index (no instruction)."""
        meta = self.bge.metadata_dict.get(track_id)
        if meta is None:
            return None
        return self._encode(self.bge._stringify_metadata(meta), is_query=False)

    def _alpha(self, turn_number: Optional[int]) -> float:
        if turn_number is None or turn_number <= 1:
            return 0.0
        return min(self.alpha_start + self.alpha_step * turn_number, self.alpha_cap)

    @staticmethod
    def _zscore(scores: torch.Tensor) -> torch.Tensor:
        std = scores.std()
        if std < 1e-8:
            return torch.zeros_like(scores)
        return (scores - scores.mean()) / std

    def _cf_query(self, positive_track_ids: Optional[List[str]]) -> Optional[torch.Tensor]:
        """Mean cf-bpr vector of accumulated positive tracks, normalized [128]. None if none have cf."""
        if not positive_track_ids:
            return None
        vecs = [self.cf_embs[self._cf_id_to_idx[t]] for t in positive_track_ids if t in self._cf_id_to_idx]
        if not vecs:
            return None
        return F.normalize(torch.stack(vecs, dim=0).mean(dim=0), p=2, dim=0)

    def text_to_item_retrieval(self, query: str, topk: int, user_id=None,
                               anchor_track_id: Optional[str] = None,
                               positive_track_ids: Optional[List[str]] = None,
                               exclude_ids: Optional[List[str]] = None,
                               turn_number: Optional[int] = None) -> List[str]:
        query_vec = self._encode(query, is_query=True)
        exclude = set(exclude_ids or [])

        # Anchor vector blending (same BGE space)
        alpha = self._alpha(turn_number)
        if alpha > 0 and anchor_track_id is not None:
            anchor_vec = self._encode_anchor(anchor_track_id)
            if anchor_vec is not None:
                query_vec = F.normalize(alpha * anchor_vec + (1 - alpha) * query_vec, p=2, dim=0)
                exclude.add(anchor_track_id)

        bge_score = self.track_embs @ query_vec   # [N]

        # cf-bpr channel (score-level z-score fusion)
        cf_q = self._cf_query(positive_track_ids) if self.beta > 0 else None
        if cf_q is not None:
            cf_raw = self.cf_embs @ cf_q          # [M]
            aligned = torch.full((len(self.track_ids),), float("nan"), device=self.device)
            aligned[self.valid_cf] = cf_raw[self.cf_pos[self.valid_cf]]
            nan_mask = torch.isnan(aligned)
            aligned[nan_mask] = aligned[~nan_mask].mean()   # tracks without cf get the mean
            final = (1 - self.beta) * self._zscore(bge_score) + self.beta * self._zscore(aligned)
        else:
            final = bge_score

        # Exclude past recommendations + anchor
        for eid in exclude:
            idx = self.track_id_to_idx.get(eid)
            if idx is not None:
                final[idx] = -1e9
        top = torch.topk(final, min(topk, final.shape[0])).indices.cpu().tolist()
        return [self.track_ids[i] for i in top]

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int, user_ids: List = None,
                                     anchor_track_ids: List = None, positive_track_ids: List = None,
                                     exclude_ids_list: List = None, turn_numbers: List = None) -> List[List[str]]:
        n = len(queries)
        anchor_track_ids = anchor_track_ids if anchor_track_ids is not None else [None] * n
        positive_track_ids = positive_track_ids if positive_track_ids is not None else [None] * n
        exclude_ids_list = exclude_ids_list if exclude_ids_list is not None else [None] * n
        turn_numbers = turn_numbers if turn_numbers is not None else [None] * n
        return [
            self.text_to_item_retrieval(
                queries[i], topk,
                anchor_track_id=anchor_track_ids[i],
                positive_track_ids=positive_track_ids[i],
                exclude_ids=exclude_ids_list[i],
                turn_number=turn_numbers[i],
            )
            for i in range(n)
        ]



class ANCHOR_BGE_BM25_CF_MODEL:
    """BGE + BM25 cumulative query + anchor (metadata vector blend) + cf-bpr (score fusion).

    - query: the cumulative conversation (retrieval_input) encoded with BGE (with QUERY_INSTRUCTION)
    - anchor: metadata of the previous (positive) recommended track, encoded with BGE, blended in
      with weight alpha and re-normalized
    - cf-bpr: mean cf-bpr vector of the accumulated (positive) tracks vs. every track's cf-bpr,
      fused at score level via z-score with weight beta
    - BM25: lexical search with the same cumulative query, fused with the dense ranking via RRF
      (sparse/dense complementarity)
    anchor/positive/exclude/turn are injected by the pipeline (batch_chat); without them it runs query-only.
    """

    def __init__(self, bge_model, bm25_model, cf_cache_dir: str = "./precomputed/reranker",
                 beta: float = 0.2, alpha_start: float = 0.25, alpha_step: float = 0.05,
                 alpha_cap: float = 0.60, bm25_topk: int = 150, dense_pool: int = 200,
                 rrf_k: int = 10, rerank_weights: Optional[Dict[str, float]] = None,
                 rerank_pool: int = 100) -> None:

        self.bge = bge_model
        self.device = bge_model.device
        self.track_ids = bge_model.track_ids
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}
        self.track_embs = bge_model.embeddings.to(self.device)   # [N, 1024], L2-normalized
        self.beta = beta
        self.alpha_start, self.alpha_step, self.alpha_cap = alpha_start, alpha_step, alpha_cap
        self.accepts_anchor = True   # marker telling batch_chat to pass anchor info

        # BM25 sparse channel (lexical match on cumulative query) + RRF fusion hyperparameters
        self.bm25 = bm25_model       # raw BM25 index (stopwords on, no stemmer)
        self.bm25_topk = bm25_topk   # number of BM25 candidates
        self.dense_pool = dense_pool # number of dense (BGE+anchor+cf) candidates fed to RRF
        self.rrf_k = rrf_k           # RRF constant K (sweep: 10 is best, nDCG@20 0.1489)

        # Rule-based reranker (no training): re-sorts the RRF top-N by metadata signal boosts.
        #  - rerank_pool: number of RRF candidates fed to the reranker (top-100).
        #  - rerank_weights: per-signal weights. None/{} disables the reranker (RRF order as-is).
        #    Best on the devset sweep was {"artist":2.0,"era":1.0,"pop":0.2} (+0.005), but blindset
        #    generalization is unverified, so it is off by default. Pass weights explicitly to enable.
        self.rerank_pool = rerank_pool
        self.rerank_weights = rerank_weights if rerank_weights is not None else {}

        # cf-bpr track index (reuses the normalized cache built by RERANKER)
        cf_embs = torch.load(os.path.join(cf_cache_dir, "cf_bpr_track.pt"), map_location="cpu")
        with open(os.path.join(cf_cache_dir, "cf_bpr_track_ids.json")) as f:
            cf_track_ids = json.load(f)
        self.cf_embs = cf_embs.to(self.device)                   # [M, 128], L2-normalized
        self._cf_id_to_idx = {tid: i for i, tid in enumerate(cf_track_ids)}
        # BGE track order -> cf index (-1 if the track has no cf embedding)
        cf_pos = [self._cf_id_to_idx.get(tid, -1) for tid in self.track_ids]
        self.cf_pos = torch.tensor(cf_pos, dtype=torch.long, device=self.device)
        self.valid_cf = self.cf_pos >= 0
        print(f"[ANCHOR_BGE_BM25_CF] BGE tracks={len(self.track_ids)} | cf-bpr aligned={int(self.valid_cf.sum())} "
              f"| beta={beta} alpha={alpha_start}->{alpha_cap} | bm25_topk={bm25_topk} dense_pool={dense_pool} rrf_k={rrf_k} "
              f"| rerank={self.rerank_weights or 'off'} pool={rerank_pool}")

    @staticmethod
    def _rrf_fuse(ranked_lists: List[List[str]], topk: int, k: int) -> List[str]:
        """Reciprocal Rank Fusion: sum 1/(k+rank) across several ranked lists and re-sort.

        - Each list gives rank r (1-based) a score of 1/(k+r); a track's scores are summed across lists.
        - Uses ranks only, not raw scores (cosine vs. BM25), so it is safe for sparse+dense fusion.
        - Tracks ranked high in both lists get the highest fused score.
        Args:
            ranked_lists: ranked track_id lists per channel (best first).
            topk: number of results to return.
            k: RRF constant. Smaller k weights top ranks more heavily. Conventionally 60.
        Returns:
            Top-k track_ids by fused score, descending.
        """
        fused_scores: dict = {}
        for ranked in ranked_lists:
            for rank, track_id in enumerate(ranked):
                # enumerate is 0-based, so rank+1 gives the 1-based rank
                fused_scores[track_id] = fused_scores.get(track_id, 0.0) + 1.0 / (k + rank + 1)
        ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [track_id for track_id, _score in ordered[:topk]]

    @torch.no_grad()
    def _encode(self, text: str, is_query: bool) -> torch.Tensor:
        """BGE CLS encoding + L2 normalization. Queries get the instruction prefix."""
        if is_query:
            text = self.bge.QUERY_INSTRUCTION + text
        batch = self.bge.tokenizer([text], padding=True, truncation=True,
                                   max_length=self.bge.max_length, return_tensors="pt").to(self.bge.device)
        emb = self.bge.model(**batch).last_hidden_state[:, 0]
        return F.normalize(emb, p=2, dim=1).squeeze(0).to(self.device)

    def _encode_anchor(self, track_id: str) -> Optional[torch.Tensor]:
        """Encode anchor track metadata in the same format as the candidate index (no instruction)."""
        meta = self.bge.metadata_dict.get(track_id)
        if meta is None:
            return None
        return self._encode(self.bge._stringify_metadata(meta), is_query=False)

    def _alpha(self, turn_number: Optional[int]) -> float:
        if turn_number is None or turn_number <= 1:
            return 0.0
        return min(self.alpha_start + self.alpha_step * turn_number, self.alpha_cap)

    @staticmethod
    def _zscore(scores: torch.Tensor) -> torch.Tensor:
        std = scores.std()
        if std < 1e-8:
            return torch.zeros_like(scores)
        return (scores - scores.mean()) / std

    def _cf_query(self, positive_track_ids: Optional[List[str]]) -> Optional[torch.Tensor]:
        """Mean cf-bpr vector of accumulated positive tracks, normalized [128]. None if none have cf."""
        if not positive_track_ids:
            return None
        vecs = [self.cf_embs[self._cf_id_to_idx[t]] for t in positive_track_ids if t in self._cf_id_to_idx]
        if not vecs:
            return None
        return F.normalize(torch.stack(vecs, dim=0).mean(dim=0), p=2, dim=0)

    def text_to_item_retrieval(self, query: str, topk: int, user_id=None,
                               anchor_track_id: Optional[str] = None,
                               positive_track_ids: Optional[List[str]] = None,
                               exclude_ids: Optional[List[str]] = None,
                               turn_number: Optional[int] = None) -> List[str]:
        query_vec = self._encode(query, is_query=True)
        exclude = set(exclude_ids or [])

        # Anchor vector blending (same BGE space)
        alpha = self._alpha(turn_number)
        if alpha > 0 and anchor_track_id is not None:
            anchor_vec = self._encode_anchor(anchor_track_id)
            if anchor_vec is not None:
                query_vec = F.normalize(alpha * anchor_vec + (1 - alpha) * query_vec, p=2, dim=0)
                exclude.add(anchor_track_id)

        bge_score = self.track_embs @ query_vec   # [N]

        # cf-bpr channel (score-level z-score fusion)
        cf_q = self._cf_query(positive_track_ids) if self.beta > 0 else None
        if cf_q is not None:
            cf_raw = self.cf_embs @ cf_q          # [M]
            aligned = torch.full((len(self.track_ids),), float("nan"), device=self.device)
            aligned[self.valid_cf] = cf_raw[self.cf_pos[self.valid_cf]]
            nan_mask = torch.isnan(aligned)
            aligned[nan_mask] = aligned[~nan_mask].mean()   # tracks without cf get the mean
            final = (1 - self.beta) * self._zscore(bge_score) + self.beta * self._zscore(aligned)
        else:
            final = bge_score

        # Exclude past recommendations + anchor (sink to -1e9 in dense score)
        for eid in exclude:
            idx = self.track_id_to_idx.get(eid)
            if idx is not None:
                final[idx] = -1e9

        # Dense (BGE+anchor+cf) candidate pool: top dense_pool tracks by score
        dense_n = min(self.dense_pool, final.shape[0])
        dense_indices = torch.topk(final, dense_n).indices.cpu().tolist()
        dense_ranked = [self.track_ids[i] for i in dense_indices if self.track_ids[i] not in exclude]

        # BM25 (sparse) candidate pool: lexical search on the same query, minus excluded tracks
        bm25_ranked = [t for t in self.bm25.text_to_item_retrieval(query, self.bm25_topk) if t not in exclude]

        # Fuse both channels with RRF -> reranker input pool (rerank_pool >= topk)
        pool = max(topk, self.rerank_pool)
        fused = self._rrf_fuse([dense_ranked, bm25_ranked], pool, self.rrf_k)

        # Rule-based reranker: if enabled, re-sort by metadata boosts; otherwise keep RRF order
        if self.rerank_weights:
            fused = self._rule_rerank(query, fused)
        return fused[:topk]

    # ----- Rule-based reranker (no training) ---------------------------------
    # Boosts the RRF top-N by metadata signals. Signals/weights were validated on a devset sweep
    # (artist/era/pop; exact_name dropped).
    @staticmethod
    def _meta_first(value) -> str:
        """Lowercased first element of a list[str] metadata field (track_name/artist_name); '' if empty."""
        if isinstance(value, list):
            return value[0].lower() if value else ""
        if isinstance(value, str):
            return value.lower()
        return ""

    def _signal_boosts(self, signal_name: str, query_lower: str,
                       candidate_ids: List[str]) -> Dict[str, float]:
        """Compute {track_id: boost} for one signal, using bge.metadata_dict (full metadata rows)."""
        meta = self.bge.metadata_dict
        boost: Dict[str, float] = {}
        if signal_name == "artist":
            # Boost if the full artist name appears in the query (>= 3 chars to avoid false matches)
            for tid in candidate_ids:
                artist = self._meta_first(meta.get(tid, {}).get("artist_name"))
                if artist and len(artist) >= 3 and artist in query_lower:
                    boost[tid] = 1.0
        elif signal_name == "exact_name":
            # Boost if the full track name appears in the query (>= 4 chars to avoid false matches)
            for tid in candidate_ids:
                name = self._meta_first(meta.get(tid, {}).get("track_name"))
                if name and len(name) >= 4 and name in query_lower:
                    boost[tid] = 1.0
        elif signal_name == "era":
            # Boost if a year (YYYY) or decade (e.g. 90s) in the query matches the release_date year
            years = set(re.findall(r"\b(?:19\d{2}|20\d{2})\b", query_lower))
            decades = {m[1] for m in re.findall(r"\b(19|20)?(\d0)s\b", query_lower)}
            if not years and not decades:
                return boost
            for tid in candidate_ids:
                release_date = meta.get(tid, {}).get("release_date")
                if not release_date:
                    continue
                year_str = str(release_date)[:4]
                if not year_str.isdigit():
                    continue
                if year_str in years:
                    boost[tid] = 1.0
                    continue
                decade_two_digit = f"{(int(year_str) % 100) // 10 * 10:02d}"
                if decade_two_digit in decades:
                    boost[tid] = 1.0
        elif signal_name == "pop":
            # Weak auxiliary boost: popularity min-max normalized within the candidate pool
            pops = np.array(
                [float(meta.get(tid, {}).get("popularity", 0) or 0) for tid in candidate_ids],
                dtype=float,
            )
            span = pops.max() - pops.min()
            if span < 1e-9:
                return {tid: 0.0 for tid in candidate_ids}
            normalized = (pops - pops.min()) / span
            return {tid: float(normalized[i]) for i, tid in enumerate(candidate_ids)}
        return boost

    def _rule_rerank(self, query: str, fused: List[str]) -> List[str]:
        """Re-sort the RRF top-N (fused) by rule-based signal boosts.

        Base score = RRF-style reciprocal of the position, 1/(rrf_k+i+1), z-score normalized to
        match the signal scale. The base alone preserves RRF order, so only the boosts reorder it.
        """
        query_lower = query.lower()
        candidate_count = len(fused)
        base_scores = np.array([1.0 / (self.rrf_k + i + 1) for i in range(candidate_count)], dtype=float)
        base_z = (base_scores - base_scores.mean()) / (base_scores.std() + 1e-8)
        score: Dict[str, float] = {tid: float(base_z[i]) for i, tid in enumerate(fused)}
        # Accumulate weighted boosts for enabled (weight > 0) signals only
        for signal_name, weight in self.rerank_weights.items():
            if weight <= 0:
                continue
            for tid, boost in self._signal_boosts(signal_name, query_lower, fused).items():
                score[tid] += weight * boost
        # Sort by score descending (stable sort keeps RRF order on ties)
        return sorted(fused, key=lambda tid: score[tid], reverse=True)

    def get_channel_rankings(self, query: str, anchor_track_id: Optional[str] = None,
                             positive_track_ids: Optional[List[str]] = None,
                             exclude_ids: Optional[List[str]] = None,
                             turn_number: Optional[int] = None):
        """Return the two channel candidate lists (dense_ranked, bm25_ranked) right before RRF.

        Used for rrf_k sweeps: run the expensive dense encoding + BM25 search once per turn and
        re-fuse (_rrf_fuse) with different rrf_k values. rrf_k does not affect candidate extraction.

        NOTE: the body must stay identical to text_to_item_retrieval up to the RRF step, so that
        re-fusing with rrf_k=60 reproduces 0.1468. Keep both in sync when editing either.
        """
        query_vec = self._encode(query, is_query=True)
        exclude = set(exclude_ids or [])

        # Anchor vector blending (same BGE space)
        alpha = self._alpha(turn_number)
        if alpha > 0 and anchor_track_id is not None:
            anchor_vec = self._encode_anchor(anchor_track_id)
            if anchor_vec is not None:
                query_vec = F.normalize(alpha * anchor_vec + (1 - alpha) * query_vec, p=2, dim=0)
                exclude.add(anchor_track_id)

        bge_score = self.track_embs @ query_vec   # [N]

        # cf-bpr channel (score-level z-score fusion)
        cf_q = self._cf_query(positive_track_ids) if self.beta > 0 else None
        if cf_q is not None:
            cf_raw = self.cf_embs @ cf_q          # [M]
            aligned = torch.full((len(self.track_ids),), float("nan"), device=self.device)
            aligned[self.valid_cf] = cf_raw[self.cf_pos[self.valid_cf]]
            nan_mask = torch.isnan(aligned)
            aligned[nan_mask] = aligned[~nan_mask].mean()   # tracks without cf get the mean
            final = (1 - self.beta) * self._zscore(bge_score) + self.beta * self._zscore(aligned)
        else:
            final = bge_score

        # Exclude past recommendations + anchor (sink to -1e9 in dense score)
        for eid in exclude:
            idx = self.track_id_to_idx.get(eid)
            if idx is not None:
                final[idx] = -1e9

        # Dense (BGE+anchor+cf) candidate pool: top dense_pool tracks by score
        dense_n = min(self.dense_pool, final.shape[0])
        dense_indices = torch.topk(final, dense_n).indices.cpu().tolist()
        dense_ranked = [self.track_ids[i] for i in dense_indices if self.track_ids[i] not in exclude]

        # BM25 (sparse) candidate pool: lexical search on the same query, minus excluded tracks
        bm25_ranked = [t for t in self.bm25.text_to_item_retrieval(query, self.bm25_topk) if t not in exclude]
        return dense_ranked, bm25_ranked

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int, user_ids: List = None,
                                     anchor_track_ids: List = None, positive_track_ids: List = None,
                                     exclude_ids_list: List = None, turn_numbers: List = None) -> List[List[str]]:
        n = len(queries)
        anchor_track_ids = anchor_track_ids if anchor_track_ids is not None else [None] * n
        positive_track_ids = positive_track_ids if positive_track_ids is not None else [None] * n
        exclude_ids_list = exclude_ids_list if exclude_ids_list is not None else [None] * n
        turn_numbers = turn_numbers if turn_numbers is not None else [None] * n
        return [
            self.text_to_item_retrieval(
                queries[i], topk,
                anchor_track_id=anchor_track_ids[i],
                positive_track_ids=positive_track_ids[i],
                exclude_ids=exclude_ids_list[i],
                turn_number=turn_numbers[i],
            )
            for i in range(n)
        ]
