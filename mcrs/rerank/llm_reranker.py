import re
from typing import Any, Dict, List, Optional


def _first_str(value: Any, default: str = "?") -> str:
    """Pull a single display string out of a metadata field.

    Track metadata fields in this dataset are stored as lists (e.g. track_name=["..."]),
    but some fields can be missing or already a scalar. This normalizes all three cases
    so format_candidates never crashes on an unexpected shape.
    """
    # Missing field (None or empty list) -> placeholder.
    if value is None:
        return default
    # List field -> take the first element if any, else placeholder.
    if isinstance(value, list):
        return str(value[0]) if len(value) > 0 else default
    # Scalar field -> stringify as-is.
    return str(value)


def _format_track(meta: Dict[str, Any], max_tags: int = 3) -> str:
    """Format one track's metadata as a single descriptive line (no leading number).

    Shared by the listwise candidate list and the pointwise single-track prompt.
    max_tags caps how many tags are shown (None = all); tags average ~34 per track.
    """
    # Core display fields (lists in this dataset) -> first value.
    name = _first_str(meta.get("track_name"))
    artist = _first_str(meta.get("artist_name"))
    # Tags: up to max_tags (None = all), comma-joined; empty string if absent.
    tag_list = meta.get("tag_list") or []
    capped_tags = tag_list if max_tags is None else tag_list[:max_tags]
    tags = ", ".join(capped_tags) if isinstance(tag_list, list) else str(tag_list)
    # Popularity / release_date are not in corpus_types -> may be missing.
    popularity = meta.get("popularity", 0)
    year = str(_first_str(meta.get("release_date"), default=""))[:4]
    return f'"{name}" — {artist} | {tags} | pop:{popularity} | {year}'


def format_candidates(candidate_ids: List[str], track_meta_dict: Dict[str, Any],
                      max_tags: int = 3) -> str:
    """Render candidates as one numbered line per track for the LLM to read (listwise).

    Output is numbered ([1], [2], ...) so the LLM replies with numbers only;
    we map numbers back to track_ids afterwards (prevents track_id hallucination).
    """
    lines: List[str] = []
    for index, track_id in enumerate(candidate_ids, start=1):
        meta = track_meta_dict.get(track_id, {})
        lines.append(f"[{index}] {_format_track(meta, max_tags=max_tags)}")
    return "\n".join(lines)


# Reranking prompt. The negative-constraint rule is the core value the rule-based
# reranker could not provide: exclude what the user explicitly rejected.
RERANK_PROMPT = """You are a music recommendation expert helping in a conversation.

User's request and conversation context:
{query}

Candidate tracks retrieved by our system:
{candidates}

Select the {topk} tracks that BEST match what the user is asking for, ordered most to least relevant.

Rules:
- If the user explicitly rejected something (an artist, genre, or style — e.g. "not X", "something different", "other than"), EXCLUDE tracks matching that rejection.
- Prioritize the specific mood, genre, era, or artist the user mentioned.
- Output ONLY numbers, comma-separated, most relevant first. Example: 3, 17, 1, 42
- Exactly {topk} numbers from the list above. No explanation, no other text."""


# Pointwise prompt: judge ONE track. We do not read the generated text — we read the
# probability the model assigns to "Yes" vs "No" at the first answer token.
POINTWISE_PROMPT = """You are a music recommendation expert helping in a conversation.

User's request and conversation context:
{query}

Candidate track:
{track}

Is this track a good recommendation for what the user is asking for right now? Consider any rejection the user made (e.g. "not X", "something different"). Answer with a single word: Yes or No."""


class LLMReranker:
    """Rerank retrieval top-N with the response-generation LLM. Config toggle + 3-way fallback.

    When disabled (config.enabled=false or type!="llm"), rerank() is a pure no-op that just
    returns candidate_ids[:topk] — the baseline retrieval order is preserved exactly.
    """

    def __init__(self, llm: Any, track_meta_dict: Dict[str, Any], config: Any) -> None:
        # config may be an OmegaConf DictConfig or a plain dict; .get works for both.
        self.config = config
        self.enabled: bool = config.get("enabled", False)
        self.type: str = config.get("type", "none")
        # Inactive path: store nothing else, rerank() short-circuits to retrieval order.
        if not self.enabled or self.type != "llm":
            self.active = False
            return
        # Active path: keep the (shared) LLM + metadata and read the knobs.
        self.active = True
        self.llm = llm                                          # reused Qwen3-8B (self.lm)
        self.meta = track_meta_dict
        self.n_candidates: int = config.get("n_candidates", 50)  # pool size to rerank
        self.topk: int = config.get("topk", 20)                  # final output size
        self.temperature: float = config.get("temperature", 0.3)  # low for consistency
        self.max_new_tokens: int = config.get("max_new_tokens", 128)
        self.max_tags: int = config.get("max_tags", 3)            # tags shown per candidate
        self.method: str = config.get("method", "listwise")       # "listwise" | "pointwise"
        self.chunk_size: int = config.get("chunk_size", 8)        # pointwise forward batch size
        # rank_blend: weight w on the LLM ranking when blending with retrieval rank (pointwise only).
        # final_rank = (1-w)*retrieval_rank + w*llm_rank. w=1.0 -> pure LLM, w=0 -> pure retrieval.
        self.rank_blend: float = config.get("rank_blend", 1.0)

    def _call_llm(self, prompt: str) -> str:
        """Single-prompt generation via the shared LLM, forcing the low rerank temperature."""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.simple_generate(
            messages, temperature=self.temperature, max_new_tokens=self.max_new_tokens
        )

    def _parse(self, output: str, n_cand: int) -> List[int]:
        """Parse '3, 17, 1, ...' into a list of 1-based indices, validated against the pool.

        Range check (1..n_cand) + dedup is what blocks hallucination: any number the LLM
        invents outside the candidate range is dropped, so the result can only contain
        positions that actually exist in the pool.
        """
        picked: List[int] = []
        for token in re.findall(r"\d+", output):
            number = int(token)
            if 1 <= number <= n_cand and number not in picked:
                picked.append(number)
        return picked

    def _finalize(self, picked_nums: List[int], candidate_ids: List[str]) -> List[str]:
        """Map picked indices -> track_ids, then top up from retrieval order to reach topk."""
        # LLM-selected tracks (1-based index -> track_id).
        result: List[str] = [candidate_ids[number - 1] for number in picked_nums]
        # If the LLM returned fewer than topk, backfill with the original retrieval order.
        if len(result) < self.topk:
            seen = set(result)
            for track_id in candidate_ids:
                if track_id not in seen:
                    result.append(track_id)
                    seen.add(track_id)
                if len(result) >= self.topk:
                    break
        return result[: self.topk]

    def rerank(self, query: str, candidate_ids: List[str], topk: Optional[int] = None) -> List[str]:
        """Main entry. Disabled -> retrieval order. Enabled -> LLM rerank with 3-way fallback."""
        topk = topk or self.topk
        # Toggle off: pure no-op, identical to baseline retrieval.
        if not self.active:
            return candidate_ids[:topk]

        cands = candidate_ids[: self.n_candidates]            # rerank the top-50 pool
        try:
            if self.method == "pointwise":
                return self._rerank_pointwise(query, cands, topk)
            return self._rerank_listwise(query, cands, topk)
        except Exception as error:
            # Fallback: any runtime error -> retrieval order (baseline is never broken).
            print(f"[LLMReranker] error: {error} → retrieval fallback")
            return candidate_ids[:topk]

    def _rerank_listwise(self, query: str, cands: List[str], topk: int) -> List[str]:
        """Listwise: LLM reads all candidates and generates a number ordering."""
        prompt = RERANK_PROMPT.format(
            query=query,
            candidates=format_candidates(cands, self.meta, max_tags=self.max_tags),
            topk=topk,
        )
        output = self._call_llm(prompt)
        picked = self._parse(output, len(cands))
        # Fallback: parsing produced nothing usable -> retrieval order.
        if not picked:
            return cands[:topk]
        return self._finalize(picked, cands)

    def _rerank_pointwise(self, query: str, cands: List[str], topk: int) -> List[str]:
        """Pointwise: score each candidate independently by p(Yes), then sort by score.

        No hallucination is possible — we only reorder the existing candidate ids by their
        relevance probability. Ties keep the original retrieval order (stable sort).
        """
        # Build one chat-templated prompt per candidate (assistant turn started, ready to answer).
        prompts: List[str] = []
        for track_id in cands:
            track_line = _format_track(self.meta.get(track_id, {}), max_tags=self.max_tags)
            text = POINTWISE_PROMPT.format(query=query, track=track_line)
            prompts.append(self.llm._apply_template([{"role": "user", "content": text}]))
        # One forward per chunk -> p(Yes) relevance score in [0, 1] for each candidate.
        scores = self.llm.score_yes_no_batch(prompts, chunk_size=self.chunk_size)

        # LLM rank position (0 = most relevant) for each candidate index.
        order_by_score = sorted(range(len(cands)), key=lambda i: (-scores[i], i))
        llm_rank = [0] * len(cands)
        for position, idx in enumerate(order_by_score):
            llm_rank[idx] = position

        # Rank-blend: combine retrieval rank (= original index i) with the LLM rank.
        # w=1.0 -> pure LLM order; w=0.3 -> mostly retrieval, gently nudged by the LLM.
        w = self.rank_blend
        blended = [(1.0 - w) * i + w * llm_rank[i] for i in range(len(cands))]
        # Sort ascending by blended rank; ties keep retrieval order (i).
        final_order = sorted(range(len(cands)), key=lambda i: (blended[i], i))
        return [cands[i] for i in final_order][:topk]
