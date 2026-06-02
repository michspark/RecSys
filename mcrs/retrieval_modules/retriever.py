import os
import json
import re
import hashlib
from typing import List, Optional
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import ClapModel, ClapProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModel
from .extract_query_comp import extract_structured_query
from .reranker import RERANKER

_PUNCT_RE = re.compile(r"[^\w\s]")

def _normalize_name(s: str) -> str:
    return _PUNCT_RE.sub(" ", s).lower().strip()


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

class QWEN3_EMBED_MODEL:
    """Pre-computed Qwen3-0.6B track embeddings (metadata + attributes) with live query encoding.

    Track embeddings are loaded from the HuggingFace dataset and cached locally.
    Query encoding uses Qwen3-0.6B mean-pooled last hidden state.
    """

    METADATA_COL   = "metadata-qwen3_embedding_0.6b"
    ATTRIBUTES_COL = "attributes-qwen3_embedding_0.6b"
    LYRICS_COL     = "lyrics-qwen3_embedding_0.6b"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        embedding_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split_types: list = ["all_tracks"],
        cache_dir: str = "./precomputed/qwen3_embed",
        device: str = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.metadata_embeddings, self.attributes_embeddings, self.track_ids = \
            self._load_or_build_index(embedding_dataset, split_types)
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}
        self.lyrics_embeddings = self._load_or_build_lyrics(embedding_dataset, split_types)

        print(f"[Qwen3Embed] Loading encoder {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.model.to(self.device).eval()
        print(f"[Qwen3Embed] Ready — {len(self.track_ids)} tracks, dim=1024")

    def _load_or_build_index(self, dataset_name, split_types):
        os.makedirs(self.cache_dir, exist_ok=True)
        meta_path  = os.path.join(self.cache_dir, "metadata.pt")
        attr_path  = os.path.join(self.cache_dir, "attributes.pt")
        ids_path   = os.path.join(self.cache_dir, "track_ids.json")

        if os.path.exists(meta_path) and os.path.exists(attr_path) and os.path.exists(ids_path):
            meta_emb = torch.load(meta_path, map_location="cpu")
            attr_emb = torch.load(attr_path, map_location="cpu")
            with open(ids_path) as f:
                track_ids = json.load(f)
            print(f"[Qwen3Embed] Loaded {len(track_ids)} tracks from cache")
            return meta_emb, attr_emb, track_ids

        print("[Qwen3Embed] Building index from HuggingFace dataset...")
        ds = load_dataset(dataset_name)
        splits = [ds[s] for s in split_types if s in ds]
        full = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

        track_ids, meta_rows, attr_rows = [], [], []
        for row in full:
            me = row[self.METADATA_COL]
            at = row[self.ATTRIBUTES_COL]
            if me and len(me) == 1024 and at and len(at) == 1024:
                track_ids.append(row["track_id"])
                meta_rows.append(me)
                attr_rows.append(at)

        meta_emb = F.normalize(torch.tensor(meta_rows, dtype=torch.float32), p=2, dim=1)
        attr_emb = F.normalize(torch.tensor(attr_rows, dtype=torch.float32), p=2, dim=1)

        torch.save(meta_emb, meta_path)
        torch.save(attr_emb, attr_path)
        with open(ids_path, "w") as f:
            json.dump(track_ids, f)
        print(f"[Qwen3Embed] Built index: {len(track_ids)} tracks")
        return meta_emb, attr_emb, track_ids

    def _load_or_build_lyrics(self, dataset_name, split_types):
        lyrics_path = os.path.join(self.cache_dir, "lyrics.pt")
        if os.path.exists(lyrics_path):
            emb = torch.load(lyrics_path, map_location="cpu")
            print(f"[Qwen3Embed] Loaded lyrics: {emb.shape[0]} tracks")
            return emb

        print("[Qwen3Embed] Building lyrics index from HuggingFace dataset...")
        ds = load_dataset(dataset_name)
        splits = [ds[s] for s in split_types if s in ds]
        full = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

        n = len(self.track_ids)
        lyrics_emb = torch.zeros(n, 1024)
        count = 0
        for row in full:
            tid = row["track_id"]
            if tid not in self.track_id_to_idx:
                continue
            ly = row.get(self.LYRICS_COL)
            if ly and len(ly) == 1024:
                lyrics_emb[self.track_id_to_idx[tid]] = torch.tensor(ly, dtype=torch.float32)
                count += 1

        lyrics_emb = F.normalize(lyrics_emb, p=2, dim=1)
        torch.save(lyrics_emb, lyrics_path)
        print(f"[Qwen3Embed] Lyrics index: {count}/{n} tracks")
        return lyrics_emb

    def encode_query(self, text: str) -> torch.Tensor:
        """Encode text with Qwen3-0.6B mean pooling → normalized [1024] vector."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            hidden = out.last_hidden_state           # [1, T, 1024]
            mask   = inputs["attention_mask"].unsqueeze(-1).float()
            emb    = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb    = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).cpu()                  # [1024]


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
        qwen3_embed: Optional[QWEN3_EMBED_MODEL] = None,
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
        clap_dim = clap_model.embeddings.shape[1]
        aligned = torch.zeros(len(self.track_ids), clap_dim)
        for i, tid in enumerate(self.track_ids):
            if tid in clap_model.track_id_to_idx:
                j = clap_model.track_id_to_idx[tid]
                aligned[i] = clap_model.embeddings[j]
        self.clap_aligned = aligned  # [N_bge, clap_dim]
        self.reranker = reranker

        # Pre-align Qwen3 metadata/attributes/lyrics embeddings to BGE track order.
        self.qwen3_embed = qwen3_embed
        if qwen3_embed is not None:
            n = len(self.track_ids)
            self.meta_aligned   = torch.zeros(n, 1024)
            self.attr_aligned   = torch.zeros(n, 1024)
            self.lyrics_aligned = torch.zeros(n, 1024)
            for i, tid in enumerate(self.track_ids):
                if tid in qwen3_embed.track_id_to_idx:
                    j = qwen3_embed.track_id_to_idx[tid]
                    self.meta_aligned[i]   = qwen3_embed.metadata_embeddings[j]
                    self.attr_aligned[i]   = qwen3_embed.attributes_embeddings[j]
                    self.lyrics_aligned[i] = qwen3_embed.lyrics_embeddings[j]
            print(f"[HYBRID] Qwen3 embeddings aligned: {n} tracks")

        # Popularity scores normalized to [0, 1], aligned to BGE track order.
        pop = torch.zeros(len(self.track_ids))
        for i, tid in enumerate(self.track_ids):
            meta = bge_model.metadata_dict.get(tid, {})
            v = meta.get("popularity")
            if v is not None:
                pop[i] = float(v)
        mx = pop.max()
        self.popularity_scores = pop / mx if mx > 0 else pop

        # Name lookup for exact-match retrieval.
        self._name_lookup = self._build_name_lookup()
        print(f"[HYBRID] Name lookup: {len(self._name_lookup)} unique track names")

    def _load_keyword_model(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        model.to(self.keyword_device).eval()
        return tokenizer, model

    def _build_name_lookup(self) -> dict:
        """Build {normalized_name: [(frozenset_artists, track_id), ...]} for exact matching."""
        lookup = {}
        for tid, meta in self.bge_model.metadata_dict.items():
            names   = meta.get("track_name", [])
            artists = meta.get("artist_name", [])
            if isinstance(names, str):   names   = [names]
            if isinstance(artists, str): artists = [artists]
            artist_set = frozenset(_normalize_name(a) for a in artists if a)
            for name in (names or []):
                if not name:
                    continue
                key = _normalize_name(name)
                if key not in lookup:
                    lookup[key] = []
                lookup[key].append((artist_set, tid))
        return lookup

    def _direct_match(self, track_name: str, artist_name: Optional[str] = None) -> Optional[str]:
        """Return track_id for an exact (normalized) title+artist match, or None."""
        candidates = self._name_lookup.get(_normalize_name(track_name))
        if not candidates:
            return None
        if artist_name:
            norm_artist = _normalize_name(artist_name)
            for artist_set, tid in candidates:
                if any(norm_artist in a or a in norm_artist for a in artist_set):
                    return tid
        return candidates[0][1]

    def _lookup_or_extract(self, query: str,
                           category: str = None, specificity: str = None) -> dict:
        """Return precomputed components if cached, otherwise run Qwen (or fallback).

        Cache key includes category+specificity so the same conversation gets
        different extractions for different categories.
        """
        key = hashlib.md5(
            f"{category or ''}_{specificity or ''}_{query}".encode("utf-8")
        ).hexdigest()
        if key in self.keyword_cache:
            return self.keyword_cache[key]
        if self.keyword_model is not None:
            lm_components = (self.keyword_model, self.keyword_tokenizer, self.keyword_device)
            return extract_structured_query(query, lm_components, category, specificity)

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

    def _meta_scores(self, query_text: str) -> torch.Tensor:
        """Qwen3 metadata embedding scores against all tracks."""
        q = self.qwen3_embed.encode_query(query_text)   # [1024]
        return torch.matmul(self.meta_aligned, q)        # [N]

    def _attr_scores(self, query_text: str) -> torch.Tensor:
        """Qwen3 attributes embedding scores against all tracks."""
        q = self.qwen3_embed.encode_query(query_text)
        return torch.matmul(self.attr_aligned, q)

    def _lyrics_scores(self, query_text: str) -> torch.Tensor:
        """Qwen3 lyrics embedding scores against all tracks."""
        q = self.qwen3_embed.encode_query(query_text)
        return torch.matmul(self.lyrics_aligned, q)

    def _common_retrieval(self, combined: torch.Tensor, components: dict,
                          topk: int, user_id) -> List[str]:
        """Shared topk → reject-filter → optional rerank → return track IDs."""
        n_cand = min(max(100, topk * 2) if self.reranker else topk * 5, combined.shape[0])
        top_result  = torch.topk(combined, k=n_cand)
        top_indices = top_result.indices.tolist()
        top_scores  = top_result.values
        candidates  = [self.track_ids[i] for i in top_indices]
        candidates  = self._filter_rejected(candidates, components["rejected"])
        if self.reranker is not None:
            score_map   = {self.track_ids[i]: top_scores[pos].item() for pos, i in enumerate(top_indices)}
            cand_scores = torch.tensor([score_map.get(tid, 0.0) for tid in candidates])
            return self.reranker.rerank(candidates, cand_scores, user_id, topk)
        return candidates[:topk]

    def _count_user_turns(self, retrieval_input: str) -> int:
        n = retrieval_input.count("\nuser:")
        if retrieval_input.startswith("user:"):
            n += 1
        return max(n, 1)

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return torch.zeros_like(scores)
        return (scores - mn) / (mx - mn)

    def _filter_rejected(self, candidates: List[str], rejected: List[str]) -> List[str]:
        if not rejected:
            return candidates
        # Qwen3가 가끔 {"name": "...", "reason": "..."} 형태의 dict를 반환함
        # 문자열만 추출해서 처리
        rej_strings = []
        for r in rejected:
            if isinstance(r, str):
                rej_strings.append(r)
            elif isinstance(r, dict):
                # name, artist, album 등 첫 번째 문자열 값을 사용
                for val in r.values():
                    if isinstance(val, str):
                        rej_strings.append(val)
                        break
        if not rej_strings:
            return candidates
        rej_lower = [r.lower() for r in rej_strings]
        filtered = [
            tid for tid in candidates
            if not any(
                rej in a.lower()
                for rej in rej_lower
                for a in (
                    [self.bge_model.metadata_dict.get(tid, {}).get("artist_name", [])]
                    if isinstance(self.bge_model.metadata_dict.get(tid, {}).get("artist_name", []), str)
                    else self.bge_model.metadata_dict.get(tid, {}).get("artist_name", [])
                )
            )
        ]
        return filtered if filtered else candidates

    # ---- category retrieval methods ----

    def _retrieve_category_a(self, query, topk, user_id, specificity):
        """Audio characteristics / mood."""
        c = self._lookup_or_extract(query, "A", specificity)
        if specificity == "HL" and self.qwen3_embed is not None:
            combined = (0.5 * self._normalize(self._meta_scores(c["bge_query"]))
                      + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"]))
                      + 0.2 * self._normalize(self._attr_scores(c["bge_query"])))
        elif specificity == "LH":
            if c.get("found"):
                # 곡을 찾은 후 → 유사곡 탐색 모드로 전환
                combined = (0.5 * self._normalize(self._meta_scores(c["bge_query"]))
                          + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"]))
                          + 0.2 * self._normalize(self._bge_scores(c["bge_query"])))
            else:
                # 아직 찾는 중 → turn-adaptive
                t = self._count_user_turns(query)
                b = self._normalize(self._bge_scores(c["bge_query"]))
                k = self._normalize(self._clap_scores(c["clap_keywords"]))
                combined = (0.6 * b + 0.4 * k) if t <= 2 else (0.4 * b + 0.6 * k)
        elif specificity == "LL" and self.qwen3_embed is not None:
            clap_q = c.get("continue_from") or c["clap_keywords"]
            combined = (0.4 * self._normalize(self._clap_scores(clap_q))
                      + 0.4 * self._normalize(self._attr_scores(c["bge_query"]))
                      + 0.2 * self._normalize(self._bge_scores(c["bge_query"])))
        else:
            combined = self._normalize(self._bge_scores(c["bge_query"]))
        return self._common_retrieval(combined, c, topk, user_id)

    def _retrieve_category_b(self, query, topk, user_id, specificity):
        """Lyrics-based search."""
        c         = self._lookup_or_extract(query, "B", specificity)
        lyrics_q  = c.get("clap_keywords") or c["bge_query"]   # build_result에서 clap_keywords = lyrics query
        bge_q     = c["bge_query"]
        phase     = c.get("phase", "searching")                 # HH 전용
        found     = c.get("found", False)                       # LH 전용

        if self.qwen3_embed is None:
            return self._common_retrieval(self._normalize(self._bge_scores(bge_q)), c, topk, user_id)

        if specificity == "HH":
            if phase == "exploring":
                # 곡 찾은 후 → 아티스트+장르 기반 탐색
                combined = (0.4 * self._normalize(self._bge_scores(bge_q))
                          + 0.3 * self._normalize(self._lyrics_scores(lyrics_q))
                          + 0.3 * self._normalize(self._meta_scores(bge_q)))
            else:
                # 가사로 찾는 중 → lyrics 집중
                combined = self._normalize(self._lyrics_scores(lyrics_q))

        elif specificity == "LH":
            if found:
                # 찾은 후 → lyrics + bge + meta 균형
                combined = (0.4 * self._normalize(self._lyrics_scores(lyrics_q))
                          + 0.3 * self._normalize(self._bge_scores(bge_q))
                          + 0.3 * self._normalize(self._meta_scores(bge_q)))
            else:
                # 아직 찾는 중 → lyrics 주도
                combined = (0.7 * self._normalize(self._lyrics_scores(lyrics_q))
                          + 0.3 * self._normalize(self._bge_scores(bge_q)))

        elif specificity == "HL":
            # 아티스트 중심 탐색: bge(아티스트 이름 매칭) + lyrics(테마)
            combined = (0.5 * self._normalize(self._bge_scores(bge_q))
                      + 0.5 * self._normalize(self._lyrics_scores(lyrics_q)))

        else:  # LL
            combined = (0.4 * self._normalize(self._lyrics_scores(lyrics_q))
                      + 0.3 * self._normalize(self._meta_scores(bge_q))
                      + 0.3 * self._normalize(self._bge_scores(bge_q)))

        return self._common_retrieval(combined, c, topk, user_id)

    def _retrieve_category_c(self, query, topk, user_id, specificity):
        """Album art / visual — image_query proxied through attr-qwen3 until SigLIP2 is wired."""
        c = self._lookup_or_extract(query, "C", specificity)
        # image_query = visual_keywords text; use attr_query when available, else raw query
        visual_q = c.get("image_query") or c.get("attr_query") or query
        bge_q    = c["bge_query"]

        if specificity == "LH" and self.qwen3_embed is not None:
            # Visual memory is primary signal; attr-qwen3 as visual proxy, BGE for artist if known
            combined = (0.6 * self._normalize(self._attr_scores(visual_q))
                      + 0.4 * self._normalize(self._bge_scores(bge_q)))
        elif specificity == "HL" and self.qwen3_embed is not None:
            # Visual style + genre both known — blend visual proxy, genre attrs, and BGE
            combined = (0.4 * self._normalize(self._attr_scores(visual_q))
                      + 0.3 * self._normalize(self._attr_scores(c.get("attr_query") or query))
                      + 0.3 * self._normalize(self._bge_scores(bge_q)))
        elif specificity == "HH" and self.qwen3_embed is not None:
            # Both specific: artist + visual + genre — BGE anchors, attrs fill
            combined = (0.5 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._attr_scores(visual_q))
                      + 0.2 * self._normalize(self._clap_scores(c["clap_keywords"])))
        elif specificity == "LL" and self.qwen3_embed is not None:
            # Vague visual + mood — CLAP for audio mood, attrs for vibe, BGE if artist known
            combined = (0.3 * self._normalize(self._clap_scores(c["clap_keywords"]))
                      + 0.4 * self._normalize(self._attr_scores(visual_q))
                      + 0.3 * self._normalize(self._bge_scores(bge_q)))
        else:
            combined = self._normalize(self._bge_scores(bge_q))
        return self._common_retrieval(combined, c, topk, user_id)

    def _retrieve_category_d(self, query, topk, user_id, specificity):
        """Exact request / themed mood / OST — routes per specificity."""
        c    = self._lookup_or_extract(query, "D", specificity)
        bge_q = c["bge_query"]

        if specificity == "HH" and self.qwen3_embed is not None:
            # Exact track request — BGE is primary for artist+title match
            combined = (0.7 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._attr_scores(c.get("attr_query") or query)))
        elif specificity == "HL" and self.qwen3_embed is not None:
            # Themed mood session — CLAP leads for sonic atmosphere, attrs reinforce
            combined = (0.4 * self._normalize(self._clap_scores(c["clap_keywords"]))
                      + 0.3 * self._normalize(self._attr_scores(c.get("attr_query") or query))
                      + 0.3 * self._normalize(self._bge_scores(bge_q)))
        elif specificity == "LH" and self.qwen3_embed is not None:
            # OST search — BGE for source/composer metadata, CLAP for emotional feel
            combined = (0.5 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"]))
                      + 0.2 * self._normalize(self._attr_scores(c.get("attr_query") or query)))
        elif specificity == "LL" and self.qwen3_embed is not None:
            # Artist exploration by vibe — attrs handle per-turn mood shift
            combined = (0.5 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"]))
                      + 0.2 * self._normalize(self._attr_scores(c.get("attr_query") or query)))
        else:
            combined = self._normalize(self._bge_scores(bge_q))
        return self._common_retrieval(combined, c, topk, user_id)

    def _retrieve_category_e(self, query, topk, user_id, specificity):
        """Technical match / musical journey / taste discovery / free exploration."""
        c     = self._lookup_or_extract(query, "E", specificity)
        bge_q = c["bge_query"]
        attr_q = c.get("attr_query") or query

        # dynamic_weights는 discovery_phase / journey_stage에 따라 cat_e가 계산해줌
        dw = c.get("dynamic_weights")

        if self.qwen3_embed is None:
            combined = self._normalize(self._bge_scores(bge_q))
        elif dw is not None:
            # E-LH / E-HL: phase/stage별 동적 가중치 적용
            combined = (dw["metadata"]   * self._normalize(self._bge_scores(bge_q))
                      + dw["attributes"] * self._normalize(self._attr_scores(attr_q))
                      + dw["audio"]      * self._normalize(self._clap_scores(c["clap_keywords"])))
        elif specificity == "HH":
            # 기술적 정확 매칭 — BGE(메타데이터)가 핵심, attr로 서브장르 보강
            combined = (0.6 * self._normalize(self._bge_scores(bge_q))
                      + 0.4 * self._normalize(self._attr_scores(attr_q)))
        elif specificity == "LL":
            # 자유 탐색 — CLAP으로 지금 원하는 소닉 무드, BGE로 좋아한 아티스트 고정
            combined = (0.3 * self._normalize(self._clap_scores(c["clap_keywords"]))
                      + 0.4 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._attr_scores(attr_q)))
        else:
            combined = self._normalize(self._bge_scores(bge_q))
        return self._common_retrieval(combined, c, topk, user_id)

    def _retrieve_category_f(self, query, topk, user_id, specificity):
        """Exact version / genre-era exploration / memory search / free genre browse."""
        c      = self._lookup_or_extract(query, "F", specificity)
        bge_q  = c["bge_query"]
        attr_q = c.get("attr_query") or query

        if self.qwen3_embed is None:
            combined = self._normalize(self._bge_scores(bge_q))
        elif specificity == "HH":
            # 정확한 앨범/버전 — BGE가 artist+album 메타데이터를 직접 매칭
            combined = (0.7 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._attr_scores(attr_q)))
        elif specificity == "HL":
            # 장르+시대 고정, 턴별 refinement — attr이 sub_refinement를 반영
            combined = (0.4 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._attr_scores(attr_q))
                      + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"])))
        elif specificity == "LH":
            # 기억 속 곡 — BGE로 누적 단서, lyrics가 있으면 lyrics-qwen3 추가
            lyrics_q = c.get("lyrics_query")
            if lyrics_q:
                combined = (0.4 * self._normalize(self._bge_scores(bge_q))
                          + 0.3 * self._normalize(self._lyrics_scores(lyrics_q))
                          + 0.2 * self._normalize(self._attr_scores(attr_q))
                          + 0.1 * self._normalize(self._clap_scores(c["clap_keywords"])))
            else:
                combined = (0.5 * self._normalize(self._bge_scores(bge_q))
                          + 0.3 * self._normalize(self._attr_scores(attr_q))
                          + 0.2 * self._normalize(self._clap_scores(c["clap_keywords"])))
        else:  # LL
            # 장르 자유 탐색 — CLAP으로 현재 방향 소닉 매칭, BGE로 장르 고정
            combined = (0.4 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"]))
                      + 0.3 * self._normalize(self._attr_scores(attr_q)))
        return self._common_retrieval(combined, c, topk, user_id)

    def _retrieve_category_g(self, query, topk, user_id, specificity):
        """Specific version / nuanced emotion / emotional memory / simple mood."""
        c      = self._lookup_or_extract(query, "G", specificity)
        bge_q  = c["bge_query"]
        attr_q = c.get("attr_query") or query

        if self.qwen3_embed is None:
            combined = self._normalize(self._bge_scores(bge_q))
        elif specificity == "HH":
            # 특정 버전/에디트 — BGE로 아티스트+장르 고정, CLAP으로 감정 톤 매칭
            combined = (0.5 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"]))
                      + 0.2 * self._normalize(self._attr_scores(attr_q)))
        elif specificity == "HL":
            # 복합 감정 탐색 — CLAP이 감정 소닉 번역을 가장 잘 잡음
            combined = (0.4 * self._normalize(self._clap_scores(c["clap_keywords"]))
                      + 0.4 * self._normalize(self._attr_scores(attr_q))
                      + 0.2 * self._normalize(self._bge_scores(bge_q)))
        elif specificity == "LH":
            # 기억 속 감정적 곡 — lyrics_query 있으면 추가, attr로 emotional 누적 클루 매칭
            lyrics_q = c.get("lyrics_query")
            if lyrics_q:
                combined = (0.3 * self._normalize(self._bge_scores(bge_q))
                          + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"]))
                          + 0.2 * self._normalize(self._attr_scores(attr_q))
                          + 0.2 * self._normalize(self._lyrics_scores(lyrics_q)))
            else:
                combined = (0.3 * self._normalize(self._bge_scores(bge_q))
                          + 0.4 * self._normalize(self._clap_scores(c["clap_keywords"]))
                          + 0.3 * self._normalize(self._attr_scores(attr_q)))
        else:  # LL
            # 단순 감정 요청 — CLAP이 무드를 직접 매칭
            lyrics_q = c.get("lyrics_query")
            if lyrics_q:
                combined = (0.3 * self._normalize(self._clap_scores(c["clap_keywords"]))
                          + 0.3 * self._normalize(self._attr_scores(attr_q))
                          + 0.2 * self._normalize(self._bge_scores(bge_q))
                          + 0.2 * self._normalize(self._lyrics_scores(lyrics_q)))
            else:
                combined = (0.4 * self._normalize(self._clap_scores(c["clap_keywords"]))
                          + 0.3 * self._normalize(self._attr_scores(attr_q))
                          + 0.3 * self._normalize(self._bge_scores(bge_q)))
        return self._common_retrieval(combined, c, topk, user_id)

    def _retrieve_category_h(self, query, topk, user_id, specificity):
        """Exact track / artist+subgenre exploration / style memory / open discovery."""
        c      = self._lookup_or_extract(query, "H", specificity)
        bge_q  = c["bge_query"]
        attr_q = c.get("attr_query") or query

        if self.qwen3_embed is None:
            combined = self._normalize(self._bge_scores(bge_q))
        elif specificity == "HH":
            if c.get("same_artist_boost"):
                # same_artist_mode: artist_name으로만 구성된 bge_query에 최대 가중치
                combined = (0.9 * self._normalize(self._bge_scores(bge_q))
                          + 0.1 * self._normalize(self._attr_scores(attr_q)))
            else:
                # 직접 요청 또는 fallback — BGE 지배
                combined = (0.7 * self._normalize(self._bge_scores(bge_q))
                          + 0.3 * self._normalize(self._attr_scores(attr_q)))
        elif specificity == "HL":
            # 아티스트/서브장르 탐색 — BGE로 앵커 고정, attr으로 현재 facet 반영
            combined = (0.5 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._attr_scores(attr_q))
                      + 0.2 * self._normalize(self._clap_scores(c["clap_keywords"])))
        elif specificity == "LH":
            # 스타일로 기억 찾기 — BGE(누적 단서) + attr + CLAP 균형
            combined = (0.4 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._attr_scores(attr_q))
                      + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"])))
        else:  # LL
            # 열린 탐색 — attr이 현재 방향, CLAP이 소닉 무드 잡음
            combined = (0.4 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._attr_scores(attr_q))
                      + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"])))
        return self._common_retrieval(combined, c, topk, user_id)

    def _retrieve_category_i(self, query, topk, user_id, specificity):
        """International/multicultural music — metadata-heavy (culture/language tags)."""
        c      = self._lookup_or_extract(query, "I", specificity)
        bge_q  = c["bge_query"]
        attr_q = c.get("attr_query") or query

        if self.qwen3_embed is None:
            combined = self._normalize(self._bge_scores(bge_q))
        elif specificity in ("HH", "HL"):
            # 구체적 요청 — BGE로 문화권+아티스트 메타데이터 직접 매칭
            combined = (0.6 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._attr_scores(attr_q))
                      + 0.1 * self._normalize(self._clap_scores(c["clap_keywords"])))
        elif specificity == "LH":
            # 기억 속 국제 히트곡 — BGE 누적 단서 + attr 보강
            combined = (0.5 * self._normalize(self._bge_scores(bge_q))
                      + 0.3 * self._normalize(self._attr_scores(attr_q))
                      + 0.2 * self._normalize(self._clap_scores(c["clap_keywords"])))
        else:  # LL
            # 문화 자유 탐색 — BGE와 attr 균등 블렌드
            combined = (0.4 * self._normalize(self._bge_scores(bge_q))
                      + 0.4 * self._normalize(self._attr_scores(attr_q))
                      + 0.2 * self._normalize(self._clap_scores(c["clap_keywords"])))
        return self._common_retrieval(combined, c, topk, user_id)

    def _retrieve_category_j(self, query, topk, user_id, specificity):
        """Popularity-aware recommendations — popularity score always active."""
        c   = self._lookup_or_extract(query, "J", specificity)
        pop = self.popularity_scores  # [N] 정규화된 popularity 스코어

        if self.qwen3_embed is None:
            combined = (0.7 * self._normalize(self._bge_scores(c["bge_query"]))
                      + 0.3 * pop)
        elif specificity == "HH":
            # 정확한 인기곡 — BGE로 아티스트+장르 고정, popularity로 순위 보강
            combined = (0.4 * self._normalize(self._bge_scores(c["bge_query"]))
                      + 0.3 * self._normalize(self._attr_scores(c.get("attr_query") or query))
                      + 0.3 * pop)
        elif specificity == "HL":
            # 니치 커뮤니티 인기 — attr이 scene 스타일 잡고 CLAP이 소닉 특성 보강
            combined = (0.3 * self._normalize(self._bge_scores(c["bge_query"]))
                      + 0.25 * self._normalize(self._attr_scores(c.get("attr_query") or query))
                      + 0.15 * self._normalize(self._clap_scores(c["clap_keywords"]))
                      + 0.3  * pop)
        elif specificity == "LH":
            # 기억 속 히트곡 — BGE 누적 단서, popularity로 실제 히트 보증
            combined = (0.4 * self._normalize(self._bge_scores(c["bge_query"]))
                      + 0.3 * self._normalize(self._attr_scores(c.get("attr_query") or query))
                      + 0.3 * pop)
        else:  # LL
            # 넓은 히트곡 브라우징 — popularity 비중 최대
            combined = (0.3 * self._normalize(self._bge_scores(c["bge_query"]))
                      + 0.3 * self._normalize(self._attr_scores(c.get("attr_query") or query))
                      + 0.4 * pop)
        return self._common_retrieval(combined, c, topk, user_id)

    def _retrieve_category_k(self, query, topk, user_id, specificity):
        """Free exploration — era/style/memory/open discovery."""
        c      = self._lookup_or_extract(query, "K", specificity)
        bge_q  = c["bge_query"]
        attr_q = c.get("attr_query") or query
        pop    = self.popularity_scores

        if self.qwen3_embed is None:
            combined = self._normalize(self._bge_scores(bge_q))
        elif specificity == "HH":
            # 연도/시대 정의 트랙 — BGE로 시대+장르 매칭, popularity로 실제 히트 확인
            if c.get("use_popularity"):
                combined = (0.4 * self._normalize(self._bge_scores(bge_q))
                          + 0.3 * self._normalize(self._attr_scores(attr_q))
                          + 0.3 * pop)
            else:
                combined = (0.5 * self._normalize(self._bge_scores(bge_q))
                          + 0.5 * self._normalize(self._attr_scores(attr_q)))
        elif specificity == "HL":
            # 스타일 앵커 탐색 — attr이 style_anchor 반영, CLAP이 소닉 방향 잡음
            combined = (0.4 * self._normalize(self._bge_scores(bge_q))
                      + 0.35 * self._normalize(self._attr_scores(attr_q))
                      + 0.25 * self._normalize(self._clap_scores(c["clap_keywords"])))
        elif specificity == "LH":
            # 기억 속 곡 — BGE(누적 단서), lyrics 있으면 추가
            lyrics_q = c.get("lyrics_query")
            if lyrics_q:
                combined = (0.4 * self._normalize(self._bge_scores(bge_q))
                          + 0.25 * self._normalize(self._attr_scores(attr_q))
                          + 0.2  * self._normalize(self._clap_scores(c["clap_keywords"]))
                          + 0.15 * self._normalize(self._lyrics_scores(lyrics_q)))
            else:
                combined = (0.45 * self._normalize(self._bge_scores(bge_q))
                          + 0.3  * self._normalize(self._attr_scores(attr_q))
                          + 0.25 * self._normalize(self._clap_scores(c["clap_keywords"])))
        else:  # LL
            # 자유 발견 — popularity_hint에 따라 pop 비중 조정
            if c.get("use_popularity"):
                combined = (0.25 * self._normalize(self._bge_scores(bge_q))
                          + 0.3  * self._normalize(self._attr_scores(attr_q))
                          + 0.2  * self._normalize(self._clap_scores(c["clap_keywords"]))
                          + 0.25 * pop)
            else:
                combined = (0.3 * self._normalize(self._bge_scores(bge_q))
                          + 0.4 * self._normalize(self._attr_scores(attr_q))
                          + 0.3 * self._normalize(self._clap_scores(c["clap_keywords"])))
        return self._common_retrieval(combined, c, topk, user_id)

    # ---- public interface ----

    def text_to_item_retrieval(self, query: str, topk: int, user_id=None,
                               category: str = None, specificity: str = None) -> List[str]:
        spec = specificity or ""
        _routes = {
            "A": self._retrieve_category_a,
            "B": self._retrieve_category_b,
            "C": self._retrieve_category_c,
            "D": self._retrieve_category_d,
            "E": self._retrieve_category_e,
            "F": self._retrieve_category_f,
            "G": self._retrieve_category_g,
            "H": self._retrieve_category_h,
            "I": self._retrieve_category_i,
            "J": self._retrieve_category_j,
            "K": self._retrieve_category_k,
        }
        fn = _routes.get(category)
        if fn:
            results = fn(query, topk, user_id, spec)
        else:
            c = self._lookup_or_extract(query)
            results = self._common_retrieval(
                self._normalize(self._bge_scores(c["bge_query"])), c, topk, user_id)

        # Global: prepend exact-match track when direct_request is detected
        c = self._lookup_or_extract(query)
        direct = c.get("direct_request")
        if direct and direct.get("track_name"):
            exact_tid = self._direct_match(direct["track_name"], direct.get("artist_name"))
            if exact_tid and exact_tid not in results[:1]:
                results = [exact_tid] + [r for r in results if r != exact_tid][:topk - 1]

        return results[:topk]

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int,
                                      user_ids: List = None,
                                      categories: List[str] = None,
                                      specificities: List[str] = None) -> List[List[str]]:
        n = len(queries)
        if user_ids      is None: user_ids      = [None] * n
        if categories    is None: categories    = [None] * n
        if specificities is None: specificities = [None] * n
        return [
            self.text_to_item_retrieval(q, topk, uid, cat, spec)
            for q, uid, cat, spec in zip(queries, user_ids, categories, specificities)
        ]
