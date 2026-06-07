import os
import json
import hashlib
from typing import List, Optional
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import ClapModel, ClapProcessor, AutoTokenizer, AutoModelForCausalLM
from .extract_query_comp import extract_structured_query
from .clap import CLAP_MODEL
from .bge import BGE_MODEL
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


class ANCHOR_CF_MODEL:
    """BGE 누적쿼리 + anchor(메타 벡터블렌드) + cf-bpr(score fusion).

    - 쿼리: 누적 대화(retrieval_input)를 BGE로 인코딩 (QUERY_INSTRUCTION 포함)
    - anchor: 직전 (긍정) 추천 트랙 메타데이터를 BGE로 인코딩해 alpha 가중합 후 재정규화
    - cf-bpr: 누적 (긍정) 트랙들의 cf-bpr 평균 ↔ 전체 트랙 cf-bpr, z-score로 beta 융합
    anchor/positive/exclude/turn 은 pipeline(batch_chat)에서 주입한다. 없으면 query-only로 동작.
    """

    def __init__(self, bge_model, cf_cache_dir: str = "./precomputed/reranker",
                 beta: float = 0.2, alpha_start: float = 0.25, alpha_step: float = 0.05,
                 alpha_cap: float = 0.60) -> None:
        self.bge = bge_model
        self.device = bge_model.device
        self.track_ids = bge_model.track_ids
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}
        self.track_embs = bge_model.embeddings.to(self.device)   # [N, 1024] 정규화됨
        self.beta = beta
        self.alpha_start, self.alpha_step, self.alpha_cap = alpha_start, alpha_step, alpha_cap
        self.accepts_anchor = True   # batch_chat이 anchor 정보를 넘길지 판단하는 마커

        # cf-bpr 트랙 인덱스 (RERANKER가 빌드한 정규화 캐시 재사용)
        cf_embs = torch.load(os.path.join(cf_cache_dir, "cf_bpr_track.pt"), map_location="cpu")
        with open(os.path.join(cf_cache_dir, "cf_bpr_track_ids.json")) as f:
            cf_track_ids = json.load(f)
        self.cf_embs = cf_embs.to(self.device)                   # [M, 128] 정규화됨
        self._cf_id_to_idx = {tid: i for i, tid in enumerate(cf_track_ids)}
        # bge 트랙 순서 → cf 인덱스 (cf 없으면 -1)
        cf_pos = [self._cf_id_to_idx.get(tid, -1) for tid in self.track_ids]
        self.cf_pos = torch.tensor(cf_pos, dtype=torch.long, device=self.device)
        self.valid_cf = self.cf_pos >= 0
        print(f"[ANCHOR_CF] BGE tracks={len(self.track_ids)} | cf-bpr aligned={int(self.valid_cf.sum())} "
              f"| beta={beta} alpha={alpha_start}->{alpha_cap}")

    @torch.no_grad()
    def _encode(self, text: str, is_query: bool) -> torch.Tensor:
        """BGE CLS 인코딩 + L2 정규화. 쿼리면 instruction 접두."""
        if is_query:
            text = self.bge.QUERY_INSTRUCTION + text
        batch = self.bge.tokenizer([text], padding=True, truncation=True,
                                   max_length=self.bge.max_length, return_tensors="pt").to(self.bge.device)
        emb = self.bge.model(**batch).last_hidden_state[:, 0]
        return F.normalize(emb, p=2, dim=1).squeeze(0).to(self.device)

    def _encode_anchor(self, track_id: str) -> Optional[torch.Tensor]:
        """anchor 트랙 메타데이터를 후보 인덱스와 동일 포맷(instruction 없음)으로 인코딩."""
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
        """누적 긍정 트랙들의 cf-bpr 평균 → 정규화 [128]. cf 있는 게 없으면 None."""
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

        # anchor 벡터 블렌딩 (같은 BGE 공간)
        alpha = self._alpha(turn_number)
        if alpha > 0 and anchor_track_id is not None:
            anchor_vec = self._encode_anchor(anchor_track_id)
            if anchor_vec is not None:
                query_vec = F.normalize(alpha * anchor_vec + (1 - alpha) * query_vec, p=2, dim=0)
                exclude.add(anchor_track_id)

        bge_score = self.track_embs @ query_vec   # [N]

        # cf-bpr 채널 (score-level z-score 융합)
        cf_q = self._cf_query(positive_track_ids) if self.beta > 0 else None
        if cf_q is not None:
            cf_raw = self.cf_embs @ cf_q          # [M]
            aligned = torch.full((len(self.track_ids),), float("nan"), device=self.device)
            aligned[self.valid_cf] = cf_raw[self.cf_pos[self.valid_cf]]
            nan_mask = torch.isnan(aligned)
            aligned[nan_mask] = aligned[~nan_mask].mean()   # cf 없는 후보는 평균 대체
            final = (1 - self.beta) * self._zscore(bge_score) + self.beta * self._zscore(aligned)
        else:
            final = bge_score

        # 과거 추천 + anchor 제외
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
