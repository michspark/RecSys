"""
Explanation-generation ablation harness.

Compares 4 generation configurations on a FIXED, frozen set of inputs so we can inspect
exactly what the challenge metric does NOT show us locally: the rendered prompt, the generated
text, and computable lexical-diversity metrics.

    | id | model                              | prompt mode | decoding |
    |----|------------------------------------|-------------|----------|
    | A  | meta-llama/Llama-3.2-1B-Instruct   | inject      | greedy   |  <- current code (baseline)
    | B  | meta-llama/Llama-3.2-1B-Instruct   | inject      | sampling |  A->B = decoding effect
    | C  | Qwen/Qwen3-8B                      | inject      | sampling |  B->C = model size effect
    | D  | Qwen/Qwen3-8B                      | instruct    | sampling |  C->D = prompt structure effect

Retrieval is FROZEN (loaded from an existing inference dump) so this experiment is about *text*,
not retrieval. No existing repo files are modified; this is a standalone driver that REUSES the
repo's loaders (chat_history_parser, MusicCatalogDB, UserProfileDB, LLAMA_MODEL) and mirrors the
challenge's own lexical-diversity definition (metrics/metrics_diversity.py).
"""

from __future__ import annotations

import os
import gc
import csv
import json
import html
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
import transformers
from datasets import load_dataset

# Challenge Distinct-N definition (pure-python, no heavy deps) — safe to import at module load.
from metrics.metrics_diversity import compute_lexical_diversity

# NOTE: the other reused repo utilities (chat_history_parser, MusicCatalogDB, UserProfileDB,
# LLAMA_MODEL) live under the `mcrs` package, whose __init__ pulls in the full retrieval stack
# (bm25s, CLAP, ...). This harness is deliberately retrieval-free, so we import those lazily inside
# the functions that need them — keeping this module importable (and its text/metric logic testable)
# even where the retrieval dependencies are not installed. We still REUSE the real functions.


# ============================================================================================
# Configuration constants
# ============================================================================================

# The 4 ablation cells. Each isolates exactly ONE variable from the row above it.
CONFIGS: List[Dict[str, Any]] = [
    {"id": "A", "model": "meta-llama/Llama-3.2-1B-Instruct", "prompt_mode": "inject",   "do_sample": False},
    {"id": "B", "model": "meta-llama/Llama-3.2-1B-Instruct", "prompt_mode": "inject",   "do_sample": True},
    {"id": "C", "model": "Qwen/Qwen3-8B",                    "prompt_mode": "inject",   "do_sample": True},
    {"id": "D", "model": "Qwen/Qwen3-8B",                    "prompt_mode": "instruct", "do_sample": True},
]

# Sampling settings held constant across B/C/D so decoding is not a moving confound between them.
TEMPERATURE: float = 0.8
TOP_P: float = 0.9

# Same generation length for ALL configs so response length is not a confound.
MAX_NEW_TOKENS: int = 128

# Fixed seed re-applied before every generate_batch call (sampling is only reproducible with it).
SEED: int = 42

# Per-model batch sizes (RTX 4090, 24GB). Lower if OOM.
BATCH_SIZE_LLAMA: int = 8
BATCH_SIZE_QWEN: int = 2

# Attention implementation per model. Qwen3-8B fills most of 24GB with weights, so we use memory-light
# "sdpa" (PyTorch scaled_dot_product_attention) instead of "eager" (which materializes the full
# attention matrix in float32 and OOMs). Llama stays on "eager" to keep config A == current code.
ATTN_LLAMA: str = "eager"
ATTN_QWEN: str = "sdpa"

# Default data locations (frozen retrieval + eval dataset).
DEFAULT_FROZEN = "exp/inference/devset/llama1b_anchor_bge_bm25_cf_devset.json"
DEFAULT_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
ITEM_DB_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
USER_DB_NAME = "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
CORPUS_TYPES = ["track_name", "artist_name", "album_name"]   # mirror config/*.yaml default
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "mcrs", "system_prompts")

# Instruction used by the "instruct" prompt mode (config D). It places the recommended track inside
# an explicit user instruction instead of pre-filling it as an assistant turn ("inject" mode).
INSTRUCT_TEMPLATE = (
    "Here is the track our system retrieved for the user:\n{recommend_item}\n\n"
    "Write a short, personalized explanation (2-3 sentences) for why this track fits what the user "
    "is looking for, grounded in the conversation above. Speak naturally; do not list raw metadata fields."
)


# ============================================================================================
# Small shim so we can reuse chat_history_parser, which expects an object exposing `.item_db`
# ============================================================================================

class _CrsShim:
    """Minimal stand-in for CRS_BASELINE: chat_history_parser only touches `.item_db.id_to_metadata`."""

    def __init__(self, item_db: MusicCatalogDB) -> None:
        self.item_db = item_db


# ============================================================================================
# System-prompt assembly (mirrors CRS_BASELINE._get_system_prompt; see crs_baseline.py:91).
# We mirror the 3-line concat instead of instantiating CRS_BASELINE, because that would load the
# LM + the full retrieval stack we are deliberately NOT using here.
# ============================================================================================

def load_role_prompts() -> Dict[str, str]:
    """Read the three system-prompt fragments from mcrs/system_prompts/."""
    return {
        "role_play": open(f"{PROMPTS_DIR}/roleplay.txt", "r", encoding="utf-8").read(),
        "personalization": open(f"{PROMPTS_DIR}/personalization.txt", "r", encoding="utf-8").read(),
        "response_generation": open(f"{PROMPTS_DIR}/response_generation.txt", "r", encoding="utf-8").read(),
    }


def build_system_prompt(role_prompt: Dict[str, str], user_db: UserProfileDB, user_id: str) -> str:
    """Construct the personalized system prompt exactly as the production pipeline does."""
    # Base = role play + response-generation instructions.
    system_prompt = role_prompt["role_play"] + role_prompt["response_generation"]
    # Append the personalization segment + the user's profile string (user_id is always present here).
    if user_id:
        user_profile_str = user_db.id_to_profile_str(user_id)
        system_prompt += role_prompt["personalization"] + "\n" + user_profile_str
    return system_prompt


# ============================================================================================
# Sample-set construction: join frozen retrieval with the eval dataset, stratify by category.
# ============================================================================================

def build_sample_inputs(
    frozen_path: str,
    dataset_name: str,
    n_per_category: int,
) -> List[Dict[str, Any]]:
    """
    Build the FIXED sample set, ~n_per_category inputs per category A-K.

    Each sample is one (session_id, turn_number) drawn from real eval data, carrying everything the
    generator needs: the system prompt, the chat history (incl. the current user turn), and the
    FROZEN top-1 recommend_item (we do NOT re-run retrieval).
    """
    # Lazy import (see module-level note): reuse the repo's real loaders/parser without paying the
    # retrieval-stack import cost unless we actually build samples.
    from run_inference_devset import chat_history_parser
    from mcrs.db_item.music_catalog import MusicCatalogDB
    from mcrs.db_user.user_profile import UserProfileDB

    # Load the frozen inference dump: list of {session_id, user_id, turn_number, predicted_track_ids, ...}.
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen_records = json.load(f)

    # Map (session_id, turn_number) -> frozen top-1 track id.
    top1_by_key: Dict[Tuple[str, int], str] = {}
    for record in frozen_records:
        key = (record["session_id"], record["turn_number"])
        predicted = record.get("predicted_track_ids") or []
        if predicted:
            top1_by_key[key] = predicted[0]

    # Load the eval dataset (conversations + category labels) and index by session_id.
    dataset = load_dataset(dataset_name, split="test")
    session_to_item: Dict[str, Dict[str, Any]] = {item["session_id"]: item for item in dataset}

    # Load DB accessors used to materialize metadata / profiles.
    item_db = MusicCatalogDB(ITEM_DB_NAME, ["all_tracks"], CORPUS_TYPES)
    user_db = UserProfileDB(USER_DB_NAME, ["all_users"])
    role_prompt = load_role_prompts()
    crs_shim = _CrsShim(item_db)

    # Group sessions by their conversation_goal.category (A-K). Sort everything for determinism.
    sessions_by_category: Dict[str, List[str]] = defaultdict(list)
    for session_id in sorted(session_to_item.keys()):
        item = session_to_item[session_id]
        category = (item.get("conversation_goal") or {}).get("category")
        if category is not None:
            sessions_by_category[category].append(session_id)

    samples: List[Dict[str, Any]] = []
    # Walk categories in alphabetical order; take up to n_per_category sessions each.
    for category in sorted(sessions_by_category.keys()):
        picked = 0
        for session_id in sessions_by_category[category]:
            if picked >= n_per_category:
                break
            item = session_to_item[session_id]
            user_id = item["user_id"]

            # Pick the turn with the MOST history available for this session, so chat_history is rich.
            candidate_turns = [
                turn for turn in range(1, 9)
                if (session_id, turn) in top1_by_key
            ]
            if not candidate_turns:
                continue  # this session never appears in the frozen dump; skip it
            target_turn = max(candidate_turns)

            # Reuse the repo's parser to build chat history up to (but excluding) the target turn.
            chat_history, user_query = chat_history_parser(item["conversations"], crs_shim, target_turn)
            # Production appends the current user query to the session memory (crs_baseline.py:160-161).
            session_memory = chat_history + [{"role": "user", "content": user_query}]

            # Frozen top-1 track -> metadata string (the shared recommend_item for all 4 configs).
            top1_track_id = top1_by_key[(session_id, target_turn)]
            recommend_item = item_db.id_to_metadata(top1_track_id)

            # Personalized system prompt, identical across configs for this input.
            sys_prompt = build_system_prompt(role_prompt, user_db, user_id)

            samples.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": target_turn,
                "category": category,
                "user_query": user_query,
                "chat_history": session_memory,   # = history + current user turn (generator input)
                "recommend_item": recommend_item,
                "sys_prompt": sys_prompt,
            })
            picked += 1

    return samples


def load_or_build_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Persist the sampled set to sample_inputs.json so the run is reproducible; reload if present."""
    sample_path = os.path.join(args.out_dir, "sample_inputs.json")
    if os.path.exists(sample_path) and not args.rebuild_samples:
        with open(sample_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"[samples] loaded {len(samples)} cached inputs from {sample_path}")
        return samples

    samples = build_sample_inputs(args.frozen, args.dataset, args.n_per_category)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    categories = sorted({s["category"] for s in samples})
    print(f"[samples] built {len(samples)} inputs across categories {categories} -> {sample_path}")
    return samples


# ============================================================================================
# ExplanationGenerator: thin wrapper that REUSES LLAMA_MODEL for loading, and adds the
# inject/instruct prompt modes + sampling + render_prompt that the ablation needs.
# ============================================================================================

class ExplanationGenerator:
    """Wraps a loaded HF model (via LLAMA_MODEL) and renders/generates under a chosen prompt mode."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 8,
        attn_implementation: str = "eager",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        # Qwen3 supports an `enable_thinking` flag in its chat template; Llama does not.
        self.is_qwen = "qwen" in model_name.lower()

        # Lazy import (see module-level note): reuse the repo's loader so model/tokenizer loading
        # stays identical to production, without dragging the retrieval stack into module import.
        from mcrs.lm_modules.llama import LLAMA_MODEL
        self._llama = LLAMA_MODEL(
            model_name=model_name, device=device,
            attn_implementation=attn_implementation, dtype=dtype,
        )
        self.lm = self._llama.lm
        self.tokenizer = self._llama.tokenizer

        # Ensure a pad token exists for left-padded batch generation (mirrors batch_response_generation).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -- prompt rendering ---------------------------------------------------------------------

    def render_prompt(
        self,
        sys_prompt: str,
        chat_history: List[Dict[str, str]],
        recommend_item: str,
        prompt_mode: str,
    ) -> str:
        """Return the exact templated string fed to the model (also shown in the report)."""
        if prompt_mode == "inject":
            # "inject" = recommend_item pre-filled as an assistant turn, generation continues from it.
            if not self.is_qwen:
                # For Llama, reuse the production formatter verbatim so config A == current code.
                return self._llama._format_chat_history(sys_prompt, chat_history, recommend_item)
            # For Qwen we replicate the same message layout but disable thinking mode.
            messages = (
                [{"role": "system", "content": sys_prompt}]
                + chat_history
                + [{"role": "assistant", "content": recommend_item}]
            )
            return self._apply_template(messages)

        if prompt_mode == "instruct":
            # "instruct" = recommend_item embedded in an explicit user instruction asking for an
            # explanation, rather than injected as a partial assistant message.
            instruction = INSTRUCT_TEMPLATE.format(recommend_item=recommend_item)
            messages = (
                [{"role": "system", "content": sys_prompt}]
                + chat_history
                + [{"role": "user", "content": instruction}]
            )
            return self._apply_template(messages)

        raise ValueError(f"unknown prompt_mode: {prompt_mode}")

    def _apply_template(self, messages: List[Dict[str, str]]) -> str:
        """apply_chat_template wrapper that passes enable_thinking=False only for Qwen."""
        kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.is_qwen:
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    # -- generation ---------------------------------------------------------------------------

    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        prompt_mode: str,
        do_sample: bool,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate one response per sample under the given prompt mode + decoding.

        Returns (responses, rendered_prompts), both aligned to `samples`.
        """
        # Render every prompt up front so we can both generate from and report on the exact strings.
        rendered_prompts = [
            self.render_prompt(s["sys_prompt"], s["chat_history"], s["recommend_item"], prompt_mode)
            for s in samples
        ]

        responses: List[str] = []
        # Process in fixed-size chunks to bound VRAM (start 8 for Llama, 4 for Qwen3-8B).
        for start in range(0, len(rendered_prompts), self.batch_size):
            chunk = rendered_prompts[start:start + self.batch_size]

            # Tokenize with left padding (tokenizer was created with padding_side="left" in LLAMA_MODEL).
            token_inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            input_ids = token_inputs.input_ids.to(self.lm.device)
            attention_mask = token_inputs.attention_mask.to(self.lm.device)

            # Assemble generate() kwargs. do_sample toggles greedy (A) vs nucleus sampling (B/C/D).
            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if do_sample:
                # temperature flattens/sharpens the next-token distribution; top_p keeps the smallest
                # set of tokens whose cumulative probability >= TOP_P (nucleus sampling).
                generate_kwargs["temperature"] = TEMPERATURE
                generate_kwargs["top_p"] = TOP_P

            with torch.no_grad():
                outputs = self.lm.generate(input_ids, attention_mask=attention_mask, **generate_kwargs)

            # Decode ONLY the newly generated tokens (strip the prompt prefix).
            decoded = self.tokenizer.batch_decode(
                outputs[:, input_ids.shape[1]:], skip_special_tokens=True
            )
            responses.extend(text.strip() for text in decoded)

        return responses, rendered_prompts

    def count_tokens(self, responses: List[str]) -> List[int]:
        """Token length of each response under THIS config's tokenizer (length sanity check)."""
        return [len(self.tokenizer(text).input_ids) for text in responses]

    def free(self) -> None:
        """Release VRAM held by this model before loading the next one."""
        del self.lm
        del self._llama
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================================
# Metrics (mirror the challenge's whitespace/lowercase Distinct-N; see metrics_diversity.py)
# ============================================================================================

def _whitespace_tokens(text: str) -> List[str]:
    """Same tokenization the challenge uses: whitespace split, no normalization."""
    return (text or "").lower().split()


def distinct_n(responses: List[str], n: int) -> float:
    """Distinct-N = unique n-grams / total n-grams, pooled across ALL responses."""
    # n=2 delegates to the challenge implementation so our scored proxy matches it exactly.
    if n == 2:
        return compute_lexical_diversity(responses, n=2)
    unique_ngrams = set()
    total_ngrams = 0
    for response in responses:
        tokens = _whitespace_tokens(response)
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            unique_ngrams.add(tuple(tokens[i:i + n]))
            total_ngrams += 1
    return len(unique_ngrams) / float(total_ngrams) if total_ngrams else 0.0


def opening_collapse(responses: List[str]) -> float:
    """(# distinct first-5-word prefixes) / N. Low => every response opens the same way (template collapse)."""
    if not responses:
        return 0.0
    prefixes = {tuple(_whitespace_tokens(text)[:5]) for text in responses}
    return len(prefixes) / float(len(responses))


def compute_metrics(responses: List[str], token_counts: List[int]) -> Dict[str, float]:
    """All 5 reported metrics over the pooled responses of one config."""
    n = max(len(responses), 1)
    return {
        "distinct_1": distinct_n(responses, 1),
        "distinct_2": distinct_n(responses, 2),                 # <- the scored-metric proxy
        "opening_collapse": opening_collapse(responses),
        "mean_tokens": sum(token_counts) / n,                   # config's own tokenizer (see caveat)
        "mean_chars": sum(len(text) for text in responses) / n,
    }


# ============================================================================================
# Report + console summary
# ============================================================================================

CAVEATS = [
    "Distinct-2 here is computed on a small (~2 per category) sample and is DIRECTIONAL ONLY; "
    "the real metric runs over the full blind set.",
    "Personalization and Explanation Quality (0.30 of the score) are NOT measured here — read the "
    "side-by-side by eye. The metrics table only covers Lexical Diversity (0.10).",
    "Sampling (configs B/C/D) is only reproducible because transformers.set_seed(42) is re-applied "
    "before every generation pass.",
    "mean_tokens is measured with each config's OWN tokenizer, so Llama vs Qwen token counts are not "
    "directly comparable; mean_chars is the tokenizer-independent length check.",
]


def write_metrics_csv(out_dir: str, metrics_by_config: Dict[str, Dict[str, float]]) -> str:
    """One row per config with all 5 metrics."""
    path = os.path.join(out_dir, "metrics.csv")
    fields = ["config", "distinct_1", "distinct_2", "opening_collapse", "mean_tokens", "mean_chars"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for config in CONFIGS:
            cid = config["id"]
            row = {"config": cid, **{k: round(v, 4) for k, v in metrics_by_config[cid].items()}}
            writer.writerow(row)
    return path


def print_metrics_table(metrics_by_config: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print the metrics table to the console."""
    header = f"{'config':<8}{'distinct_1':>12}{'distinct_2':>12}{'open_collapse':>15}{'mean_tokens':>13}{'mean_chars':>12}"
    print("\n" + header)
    print("-" * len(header))
    for config in CONFIGS:
        m = metrics_by_config[config["id"]]
        print(f"{config['id']:<8}{m['distinct_1']:>12.4f}{m['distinct_2']:>12.4f}"
              f"{m['opening_collapse']:>15.4f}{m['mean_tokens']:>13.2f}{m['mean_chars']:>12.2f}")


def _render_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Render a conversation as escaped HTML rows."""
    rows = []
    for message in chat_history:
        role = html.escape(message.get("role", ""))
        content = html.escape(message.get("content", ""))
        rows.append(f'<div class="msg"><span class="role">{role}</span>: {content}</div>')
    return "\n".join(rows)


def write_report_html(
    out_dir: str,
    samples: List[Dict[str, Any]],
    responses_by_config: Dict[str, List[str]],
    rendered_by_config: Dict[str, List[str]],
    metrics_by_config: Dict[str, Dict[str, float]],
) -> str:
    """Side-by-side report grouped BY INPUT, so the same query shows all 4 configs together."""
    path = os.path.join(out_dir, "report.html")
    parts: List[str] = []

    parts.append("""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Explanation Ablation</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;line-height:1.5;color:#222}
 .caveats{background:#fff8e1;border:1px solid #ffe082;padding:1rem 1.25rem;border-radius:8px}
 .input{border:1px solid #ddd;border-radius:8px;padding:1rem 1.25rem;margin:1.5rem 0}
 .conv{background:#f5f7fa;padding:.75rem;border-radius:6px;font-size:.92rem}
 .msg .role{font-weight:600;color:#3367d6}
 .rec{background:#eef6ee;padding:.5rem .75rem;border-radius:6px;margin:.5rem 0;font-size:.9rem}
 .cfg{border-top:1px solid #eee;padding:.75rem 0}
 .cfg h4{margin:.25rem 0}
 .out{background:#fbfbfb;border-left:3px solid #3367d6;padding:.6rem .8rem;white-space:pre-wrap}
 details{margin:.4rem 0}
 summary{cursor:pointer;color:#666;font-size:.85rem}
 pre{white-space:pre-wrap;background:#f0f0f0;padding:.6rem;border-radius:6px;font-size:.8rem}
 table{border-collapse:collapse;margin:1rem 0}
 th,td{border:1px solid #ccc;padding:.4rem .7rem;text-align:right}
 th:first-child,td:first-child{text-align:left}
</style></head><body>""")

    parts.append("<h1>Explanation-Generation Ablation</h1>")

    # Caveats banner.
    parts.append('<div class="caveats"><b>Read this first:</b><ul>')
    for caveat in CAVEATS:
        parts.append(f"<li>{html.escape(caveat)}</li>")
    parts.append("</ul></div>")

    # Metrics table.
    parts.append("<h2>Diversity metrics (Lexical Diversity proxy only)</h2><table>")
    parts.append("<tr><th>config</th><th>distinct_1</th><th>distinct_2</th><th>opening_collapse</th>"
                 "<th>mean_tokens</th><th>mean_chars</th></tr>")
    config_label = {c["id"]: f"{c['id']} ({c['model'].split('/')[-1]}, {c['prompt_mode']}, "
                             f"{'sampling' if c['do_sample'] else 'greedy'})" for c in CONFIGS}
    for config in CONFIGS:
        m = metrics_by_config[config["id"]]
        parts.append(
            f"<tr><td>{html.escape(config_label[config['id']])}</td>"
            f"<td>{m['distinct_1']:.4f}</td><td>{m['distinct_2']:.4f}</td>"
            f"<td>{m['opening_collapse']:.4f}</td><td>{m['mean_tokens']:.2f}</td>"
            f"<td>{m['mean_chars']:.2f}</td></tr>"
        )
    parts.append("</table>")

    # Per-input side-by-side.
    parts.append("<h2>Side-by-side by input</h2>")
    for index, sample in enumerate(samples):
        parts.append('<div class="input">')
        parts.append(f"<h3>Input #{index + 1} — category {html.escape(sample['category'])} "
                     f"(session {html.escape(sample['session_id'][:8])}, turn {sample['turn_number']})</h3>")
        parts.append(f'<div class="conv">{_render_chat_history(sample["chat_history"])}</div>')
        parts.append(f'<div class="rec"><b>recommend_item (frozen):</b> {html.escape(sample["recommend_item"])}</div>')

        for config in CONFIGS:
            cid = config["id"]
            response = responses_by_config[cid][index]
            rendered = rendered_by_config[cid][index]
            parts.append('<div class="cfg">')
            parts.append(f"<h4>{html.escape(config_label[cid])}</h4>")
            parts.append(f'<div class="out">{html.escape(response)}</div>')
            parts.append(f"<details><summary>rendered prompt</summary><pre>{html.escape(rendered)}</pre></details>")
            parts.append("</div>")
        parts.append("</div>")

    parts.append("</body></html>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def print_takeaway(metrics_by_config: Dict[str, Dict[str, float]]) -> None:
    """4-line console takeaway: the three pairwise Distinct-2 deltas + a length-confound flag."""
    d2 = {cid: metrics_by_config[cid]["distinct_2"] for cid in ("A", "B", "C", "D")}
    print("\n=== Takeaway ===")
    print(f"A->B (decoding): Distinct-2 {d2['A']:.4f} -> {d2['B']:.4f}  (delta {d2['B'] - d2['A']:+.4f})")
    print(f"B->C (model)   : Distinct-2 {d2['B']:.4f} -> {d2['C']:.4f}  (delta {d2['C'] - d2['B']:+.4f})")
    print(f"C->D (prompt)  : Distinct-2 {d2['C']:.4f} -> {d2['D']:.4f}  (delta {d2['D'] - d2['C']:+.4f})")

    # Length-confound flag: compare each config's mean_tokens against the overall mean.
    tokens = {cid: metrics_by_config[cid]["mean_tokens"] for cid in ("A", "B", "C", "D")}
    overall_mean = sum(tokens.values()) / len(tokens)
    flagged = [cid for cid, value in tokens.items()
               if overall_mean > 0 and abs(value - overall_mean) / overall_mean > 0.25]
    if flagged:
        print(f"[LENGTH CONFOUND] configs {flagged} mean_tokens differ >25% from the mean "
              f"({overall_mean:.1f}); diversity differences may be partly length-driven.")
    else:
        print("[length check] no config's mean_tokens differs >25% from the mean.")


# ============================================================================================
# Driver
# ============================================================================================

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Build (or reload) the fixed, frozen sample set ONCE; reused for every config.
    samples = load_or_build_samples(args)

    responses_by_config: Dict[str, List[str]] = {}
    rendered_by_config: Dict[str, List[str]] = {}
    token_counts_by_config: Dict[str, List[int]] = {}

    # Group configs by model so each model is loaded exactly once (sequential, VRAM-bounded).
    configs_by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for config in CONFIGS:
        configs_by_model[config["model"]].append(config)

    # Process models in first-appearance order: Llama (A,B) then Qwen (C,D).
    seen_models: List[str] = []
    for config in CONFIGS:
        if config["model"] not in seen_models:
            seen_models.append(config["model"])

    for model_name in seen_models:
        is_qwen = "qwen" in model_name.lower()
        # Qwen3-8B: smaller batch + memory-light sdpa attention to fit 24GB (see ATTN_QWEN note).
        batch_size = args.qwen_batch_size if is_qwen else args.llama_batch_size
        attn_implementation = ATTN_QWEN if is_qwen else ATTN_LLAMA
        print(f"\n[load] {model_name} (batch_size={batch_size}, attn={attn_implementation})")
        generator = ExplanationGenerator(
            model_name=model_name, device=args.device, batch_size=batch_size,
            attn_implementation=attn_implementation,
        )

        for config in configs_by_model[model_name]:
            cid = config["id"]
            # Re-seed before EVERY generation pass (sampling reproducibility + greedy determinism).
            transformers.set_seed(SEED)
            print(f"[run] config {cid}: mode={config['prompt_mode']} "
                  f"decoding={'sampling' if config['do_sample'] else 'greedy'}")
            responses, rendered = generator.generate_batch(
                samples, prompt_mode=config["prompt_mode"], do_sample=config["do_sample"]
            )
            responses_by_config[cid] = responses
            rendered_by_config[cid] = rendered
            token_counts_by_config[cid] = generator.count_tokens(responses)

        # Free VRAM before loading the next model.
        generator.free()
        print(f"[free] released {model_name}")

    # 2) Metrics per config over the pooled responses.
    metrics_by_config = {
        cid: compute_metrics(responses_by_config[cid], token_counts_by_config[cid])
        for cid in responses_by_config
    }

    # 3) Outputs.
    csv_path = write_metrics_csv(args.out_dir, metrics_by_config)
    report_path = write_report_html(
        args.out_dir, samples, responses_by_config, rendered_by_config, metrics_by_config
    )
    print_metrics_table(metrics_by_config)
    print(f"\n[out] metrics -> {csv_path}")
    print(f"[out] report  -> {report_path}")

    # 4) Console takeaway.
    print_takeaway(metrics_by_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explanation-generation ablation (A/B/C/D).")
    parser.add_argument("--frozen", type=str, default=DEFAULT_FROZEN,
                        help="Frozen inference dump providing the top-1 recommend_item per input.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="HF eval dataset name (provides conversations + category labels).")
    parser.add_argument("--out_dir", type=str, default="exp/explanation_ablation",
                        help="Directory for sample_inputs.json, metrics.csv, report.html.")
    parser.add_argument("--n_per_category", type=int, default=2,
                        help="Inputs sampled per category A-K (~2 -> ~22 total).")
    parser.add_argument("--llama_batch_size", type=int, default=BATCH_SIZE_LLAMA,
                        help="Batch size for the Llama-1B phase (configs A,B).")
    parser.add_argument("--qwen_batch_size", type=int, default=BATCH_SIZE_QWEN,
                        help="Batch size for the Qwen3-8B phase (configs C,D). Lower to 1 if OOM.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rebuild_samples", action="store_true",
                        help="Force rebuilding sample_inputs.json even if it already exists.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
