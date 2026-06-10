"""
Explanation-generation COMPARISON: baseline(A) vs D vs E.

Isolates ONE question: once we are on the upgraded stack (Qwen3-8B + sampling + instruct prompt
structure), does cleaning up the *system prompt* (checklist -> natural tone) make the generated
blurb read more naturally?

    | id | model                              | prompt mode | decoding | system prompt | role                         |
    |----|------------------------------------|-------------|----------|---------------|------------------------------|
    | A  | meta-llama/Llama-3.2-1B-Instruct   | inject      | greedy   | checklist     | current baseline             |
    | D  | Qwen/Qwen3-8B                      | instruct    | sampling | checklist     | upgraded stack               |
    | E  | Qwen/Qwen3-8B                      | instruct    | sampling | natural       | upgraded stack + clean prompt|

A->D = full-stack upgrade (model + decoding + prompt structure).
D->E = system-prompt CONTENT only -> D and E must be identical except for the system prompt.

This is a standalone driver. It does NOT modify run_explanation_ablation.py; it REUSES that
module's ExplanationGenerator, load_or_build_samples, the metric functions, and the data-path
constants. The only behavioural addition is a `system_prompt_key` argument on generate_batch,
provided here via a thin subclass so the original generator stays untouched.
"""

from __future__ import annotations

import os
import csv
import datetime
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import transformers

# Reuse everything that already works in the ablation harness (that file is NOT modified).
from run_explanation_ablation import (
    ExplanationGenerator,
    load_or_build_samples,
    load_role_prompts,
    compute_metrics,
    USER_DB_NAME,
    DEFAULT_FROZEN,
    DEFAULT_DATASET,
    TEMPERATURE,
    TOP_P,
    MAX_NEW_TOKENS,
    SEED,
    ATTN_LLAMA,
    ATTN_QWEN,
)


# ============================================================================================
# Configuration constants (this experiment's 3 cells: A, D, E)
# ============================================================================================

# Each cell carries which system-prompt variant it reads (system_prompt_key) plus a human label.
CONFIGS_COMPARE: List[Dict[str, Any]] = [
    {"id": "A", "model": "meta-llama/Llama-3.2-1B-Instruct", "prompt_mode": "inject",
     "do_sample": False, "system_prompt_key": "sys_prompt_checklist",
     "label": "Llama-1B, inject, greedy, checklist"},
    {"id": "D", "model": "Qwen/Qwen3-8B", "prompt_mode": "instruct",
     "do_sample": True, "system_prompt_key": "sys_prompt_checklist",
     "label": "Qwen3-8B, instruct, sampling, checklist"},
    {"id": "E", "model": "Qwen/Qwen3-8B", "prompt_mode": "instruct",
     "do_sample": True, "system_prompt_key": "sys_prompt_natural",
     "label": "Qwen3-8B, instruct, sampling, NATURAL"},
]

# Per-model batch sizes for this run (RTX 4090, 24GB): Llama 8, Qwen3-8B 4.
BATCH_SIZE_LLAMA: int = 8
BATCH_SIZE_QWEN: int = 4

# The "natural" replacement for the response_generation block. Kept in English (the conversations
# and the judge are English). This is the ONLY thing that differs between configs D and E.
NATURAL_RESPONSE_PROMPT: str = (
    "The recommendation system has already chosen a track for you. Your only job is to introduce "
    "that track to the user in a warm, natural way, the way a friend with great music taste would.\n\n"
    "Write 2-3 sentences that:\n"
    "- connect the track to something specific the user said earlier in the conversation\n"
    "- describe what it sounds or feels like in plain, everyday language\n"
    "- read like a real person talking, not a chatbot\n\n"
    "Do not gush (\"I'm so glad!\", \"perfect!\"), do not list metadata fields, and do not end by asking "
    "whether they want more recommendations. Just talk."
)

# Field order produced by MusicCatalogDB.id_to_metadata (track_id + CORPUS_TYPES). Used as
# parse boundaries so values that themselves contain commas (e.g. multiple artists) stay intact.
KNOWN_METADATA_FIELDS: Tuple[str, ...] = ("track_id", "track_name", "artist_name", "album_name")


# ============================================================================================
# System-prompt assembly: build the "natural" variant by swapping ONLY the middle block.
# (The "checklist" variant is reused verbatim from each sample's already-built sys_prompt.)
# ============================================================================================

def build_natural_system_prompt(role_prompt: Dict[str, str], user_db: Any, user_id: str) -> str:
    """Mirror build_system_prompt but replace the response_generation block with NATURAL_RESPONSE_PROMPT.

    Args:
        role_prompt: The three system-prompt fragments from load_role_prompts().
        user_db: A UserProfileDB instance used to fetch the user's profile string.
        user_id: The user identifier (always present for our samples).
    Returns:
        The full personalized system prompt using the natural-tone instruction.
    """
    # Base = role play + the NATURAL instruction (instead of the checklist response_generation block).
    system_prompt = role_prompt["role_play"] + NATURAL_RESPONSE_PROMPT
    # Append the personalization segment + the user's profile string, exactly like production.
    if user_id:
        user_profile_str = user_db.id_to_profile_str(user_id)
        system_prompt += "\n" + role_prompt["personalization"] + "\n" + user_profile_str
    return system_prompt


def attach_system_prompt_variants(samples: List[Dict[str, Any]]) -> None:
    """Add sys_prompt_checklist and sys_prompt_natural to every sample (in place).

    sys_prompt_checklist reuses the already-built sample["sys_prompt"] (production prompt).
    sys_prompt_natural rebuilds it with the natural-tone block swapped in.
    """
    # Lazy import: UserProfileDB lives under mcrs, whose __init__ pulls in the retrieval stack.
    from mcrs.db_user.user_profile import UserProfileDB

    role_prompt = load_role_prompts()
    user_db = UserProfileDB(USER_DB_NAME, ["all_users"])

    for sample in samples:
        # Checklist variant = the production system prompt that was built when sampling.
        sample["sys_prompt_checklist"] = sample["sys_prompt"]
        # Natural variant = same prompt with only the middle instruction block replaced.
        sample["sys_prompt_natural"] = build_natural_system_prompt(
            role_prompt, user_db, sample["user_id"]
        )


# ============================================================================================
# Generator subclass: add a `system_prompt_key` arg WITHOUT modifying the original generator.
# ============================================================================================

class ComparePromptGenerator(ExplanationGenerator):
    """ExplanationGenerator that can read a chosen system-prompt key per generation pass."""

    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        prompt_mode: str,
        do_sample: bool,
        system_prompt_key: str = "sys_prompt",
    ) -> Tuple[List[str], List[str]]:
        """Same as the parent, but render each prompt from samples[i][system_prompt_key].

        We build a shallow per-sample view whose "sys_prompt" points at the requested variant,
        then delegate to the unchanged parent generate_batch — so rendering/generation logic is
        reused exactly, and the original samples are not mutated.
        """
        prompt_view = [
            {**sample, "sys_prompt": sample[system_prompt_key]}
            for sample in samples
        ]
        return super().generate_batch(prompt_view, prompt_mode=prompt_mode, do_sample=do_sample)


# ============================================================================================
# Rendering helpers for the plain-text report
# ============================================================================================

def parse_metadata_fields(recommend_item: str) -> Dict[str, str]:
    """Parse 'track_id: .., track_name: .., artist_name: .., album_name: ..' into a dict.

    Splits on the KNOWN field names so commas inside a value (e.g. multiple artists) are kept.
    """
    import re

    fields: Dict[str, str] = {}
    # Lookahead boundary = the start of any next known field, or end-of-string.
    boundary = "|".join(rf"{name}: " for name in KNOWN_METADATA_FIELDS)
    for field_name in KNOWN_METADATA_FIELDS:
        # Value = text after 'field_name: ' up to ', <next known key>: ' or the end.
        pattern = rf"{field_name}: (.*?)(?:, (?:{boundary})|$)"
        match = re.search(pattern, recommend_item)
        if match:
            fields[field_name] = match.group(1).strip()
    return fields


def render_track_line(recommend_item: str) -> str:
    """Render the frozen recommend_item as a clean one-liner: 'name — artist (album: ...)'."""
    fields = parse_metadata_fields(recommend_item)
    track_name = fields.get("track_name", "?")
    artist_name = fields.get("artist_name", "?")
    album_name = fields.get("album_name")
    line = f"{track_name} — {artist_name}"
    if album_name:
        line += f"  (album: {album_name})"
    return line


def render_conversation(
    chat_history: List[Dict[str, str]],
    show_history_turns: int,
    show_full_history: bool,
) -> Tuple[str, str]:
    """Render the conversation for the report (the MODEL always gets the full history).

    Returns (header_note, body). The last user turn (current query) is always the last message
    and is therefore always included; we never truncate it away.
    """
    total = len(chat_history)
    effective_k = max(show_history_turns, 1)  # always keep at least the current query

    if show_full_history or total <= effective_k:
        shown = chat_history
        hidden = 0
        header_note = f"full history fed to the model; showing all {total} messages"
    else:
        shown = chat_history[-effective_k:]
        hidden = total - len(shown)
        header_note = f"full history fed to the model; showing last {len(shown)} of {total} messages"

    lines: List[str] = []
    if hidden:
        lines.append(f"... ({hidden} earlier message(s) hidden) ...")
    for message in shown:
        role = message.get("role", "")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return header_note, "\n".join(lines)


# ============================================================================================
# Report (report.txt is the primary artifact) + metrics.csv
# ============================================================================================

CAVEATS_COMPARE: List[str] = [
    "Distinct-2 is on a small (~2/category) sample: directional only.",
    "Personalization & explanation quality (0.30) are judged by eye here, not measured.",
    "D and E differ ONLY in the system prompt.",
]

# Metric columns shown in the report's DIVERSITY METRICS table (tokenizer-independent length only).
REPORT_METRIC_COLUMNS: List[str] = ["distinct_1", "distinct_2", "opening_collapse", "mean_chars"]


def write_report_txt(
    out_dir: str,
    samples: List[Dict[str, Any]],
    responses_by_config: Dict[str, List[str]],
    metrics_by_config: Dict[str, Dict[str, float]],
    checklist_template: str,
    natural_template: str,
    show_history_turns: int,
    show_full_history: bool,
) -> str:
    """Write the human-readable plain-text comparison report (UTF-8)."""
    path = os.path.join(out_dir, "report.txt")
    today = datetime.date.today().isoformat()
    lines: List[str] = []

    # -- Title block --------------------------------------------------------------------------
    lines.append("EXPLANATION COMPARISON — baseline(A) vs D vs E")
    lines.append(f"generated: {today} | {len(samples)} samples")
    lines.append("A->D = full-stack upgrade | D->E = system-prompt content only (the variable under test)")
    lines.append("")

    # -- Caveats ------------------------------------------------------------------------------
    lines.append("== CAVEATS ==")
    for caveat in CAVEATS_COMPARE:
        lines.append(f"- {caveat}")
    lines.append("")

    # -- The two system prompts, printed ONCE (per-user profile string omitted; appended at runtime) --
    lines.append("== SYSTEM PROMPT [checklist] (used by A & D) ==")
    lines.append("(per-user profile string appended at generation time; omitted here)")
    lines.append(checklist_template)
    lines.append("")
    lines.append("== SYSTEM PROMPT [natural] (used by E) ==")
    lines.append("(per-user profile string appended at generation time; omitted here)")
    lines.append(natural_template)
    lines.append("")

    # -- Diversity metrics table --------------------------------------------------------------
    lines.append("== DIVERSITY METRICS ==")
    header = (f"{'config':<9}{'distinct_1':>12}{'distinct_2':>12}"
              f"{'opening_collapse':>19}{'mean_chars':>12}")
    lines.append(header)
    for config in CONFIGS_COMPARE:
        metric = metrics_by_config[config["id"]]
        lines.append(
            f"{config['id']:<9}{metric['distinct_1']:>12.4f}{metric['distinct_2']:>12.4f}"
            f"{metric['opening_collapse']:>19.4f}{metric['mean_chars']:>12.2f}"
        )
    lines.append("")

    # -- Per-sample side-by-side --------------------------------------------------------------
    lines.append("== SAMPLES ==")
    bar = "#" * 80
    for index, sample in enumerate(samples):
        lines.append(bar)
        lines.append(
            f"INPUT #{index + 1}  |  category {sample['category']}  |  "
            f"session {sample['session_id'][:8]}  |  turn {sample['turn_number']}"
        )
        lines.append(bar)
        lines.append("")

        header_note, conversation_body = render_conversation(
            sample["chat_history"], show_history_turns, show_full_history
        )
        lines.append(f"[CONVERSATION]   ({header_note})")
        lines.append(conversation_body)
        lines.append("")

        lines.append("[RETRIEVED TRACK (frozen)]")
        lines.append(render_track_line(sample["recommend_item"]))
        lines.append("")

        for config in CONFIGS_COMPARE:
            cid = config["id"]
            lines.append(f"--- {cid} | {config['label']} ---")
            lines.append(responses_by_config[cid][index].strip())
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_metrics_csv(out_dir: str, metrics_by_config: Dict[str, Dict[str, float]]) -> str:
    """One row per config with all 5 metrics (same set as the ablation harness)."""
    path = os.path.join(out_dir, "metrics.csv")
    fields = ["config", "distinct_1", "distinct_2", "opening_collapse", "mean_tokens", "mean_chars"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for config in CONFIGS_COMPARE:
            cid = config["id"]
            row = {"config": cid, **{k: round(v, 4) for k, v in metrics_by_config[cid].items()}}
            writer.writerow(row)
    return path


def print_takeaway(metrics_by_config: Dict[str, Dict[str, float]]) -> None:
    """Console summary: A->D and D->E Distinct-2 deltas, plus a mean_chars length-confound flag."""
    d2 = {cid: metrics_by_config[cid]["distinct_2"] for cid in ("A", "D", "E")}
    print("\n=== Takeaway ===")
    print(f"A->D (full-stack upgrade) : Distinct-2 {d2['A']:.4f} -> {d2['D']:.4f}  (delta {d2['D'] - d2['A']:+.4f})")
    print(f"D->E (system-prompt only) : Distinct-2 {d2['D']:.4f} -> {d2['E']:.4f}  (delta {d2['E'] - d2['D']:+.4f})")

    # Length confound judged on mean_chars (tokenizer-independent, so Llama vs Qwen is comparable).
    chars = {cid: metrics_by_config[cid]["mean_chars"] for cid in ("A", "D", "E")}
    overall_mean = sum(chars.values()) / len(chars)
    flagged = [cid for cid, value in chars.items()
               if overall_mean > 0 and abs(value - overall_mean) / overall_mean > 0.25]
    if flagged:
        print(f"[LENGTH CONFOUND] configs {flagged} mean_chars differ >25% from the mean "
              f"({overall_mean:.1f}); diversity differences may be partly length-driven.")
    else:
        print("[length check] no config's mean_chars differs >25% from the mean.")


# ============================================================================================
# Driver
# ============================================================================================

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Build (or reload) the fixed, frozen sample set, then attach both system-prompt variants.
    samples = load_or_build_samples(args)
    attach_system_prompt_variants(samples)

    # Templates for the report header (full prompt minus the per-user profile, which varies).
    role_prompt = load_role_prompts()
    checklist_template = (
        role_prompt["role_play"] + role_prompt["response_generation"] + role_prompt["personalization"]
    )
    natural_template = (
        role_prompt["role_play"] + NATURAL_RESPONSE_PROMPT + "\n" + role_prompt["personalization"]
    )

    responses_by_config: Dict[str, List[str]] = {}
    rendered_by_config: Dict[str, List[str]] = {}
    token_counts_by_config: Dict[str, List[int]] = {}

    # Group configs by model so each model is loaded exactly once (sequential, VRAM-bounded).
    configs_by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for config in CONFIGS_COMPARE:
        configs_by_model[config["model"]].append(config)

    # Process models in first-appearance order: Llama (A) then Qwen (D, E reuse one load).
    seen_models: List[str] = []
    for config in CONFIGS_COMPARE:
        if config["model"] not in seen_models:
            seen_models.append(config["model"])

    for model_name in seen_models:
        is_qwen = "qwen" in model_name.lower()
        batch_size = args.qwen_batch_size if is_qwen else args.llama_batch_size
        attn_implementation = ATTN_QWEN if is_qwen else ATTN_LLAMA
        print(f"\n[load] {model_name} (batch_size={batch_size}, attn={attn_implementation})")
        generator = ComparePromptGenerator(
            model_name=model_name, device=args.device, batch_size=batch_size,
            attn_implementation=attn_implementation,
        )

        # D and E reuse this one loaded Qwen model; only system_prompt_key changes between them.
        for config in configs_by_model[model_name]:
            cid = config["id"]
            transformers.set_seed(SEED)  # re-seed before EVERY pass (sampling reproducibility)
            print(f"[run] config {cid}: mode={config['prompt_mode']} "
                  f"decoding={'sampling' if config['do_sample'] else 'greedy'} "
                  f"system_prompt={config['system_prompt_key']}")
            responses, rendered = generator.generate_batch(
                samples,
                prompt_mode=config["prompt_mode"],
                do_sample=config["do_sample"],
                system_prompt_key=config["system_prompt_key"],
            )
            responses_by_config[cid] = responses
            rendered_by_config[cid] = rendered
            token_counts_by_config[cid] = generator.count_tokens(responses)

        generator.free()
        print(f"[free] released {model_name}")

    # 2) Metrics per config over the pooled responses.
    metrics_by_config = {
        cid: compute_metrics(responses_by_config[cid], token_counts_by_config[cid])
        for cid in responses_by_config
    }

    # 3) Outputs: report.txt (primary) + metrics.csv (convenience).
    report_path = write_report_txt(
        args.out_dir, samples, responses_by_config, metrics_by_config,
        checklist_template, natural_template,
        args.show_history_turns, args.show_full_history,
    )
    csv_path = write_metrics_csv(args.out_dir, metrics_by_config)
    print(f"\n[out] report  -> {report_path}")
    print(f"[out] metrics -> {csv_path}")

    # 4) Console takeaway.
    print_takeaway(metrics_by_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explanation-generation comparison (A vs D vs E).")
    parser.add_argument("--frozen", type=str, default=DEFAULT_FROZEN,
                        help="Frozen inference dump providing the top-1 recommend_item per input.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="HF eval dataset name (provides conversations + category labels).")
    parser.add_argument("--out_dir", type=str, default="exp/explanation_compare",
                        help="Directory for sample_inputs.json, report.txt, metrics.csv.")
    parser.add_argument("--n_per_category", type=int, default=2,
                        help="Inputs sampled per category A-K (~2 -> ~22 total).")
    parser.add_argument("--llama_batch_size", type=int, default=BATCH_SIZE_LLAMA,
                        help="Batch size for the Llama-1B phase (config A).")
    parser.add_argument("--qwen_batch_size", type=int, default=BATCH_SIZE_QWEN,
                        help="Batch size for the Qwen3-8B phase (configs D, E). Lower to 2 if OOM.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rebuild_samples", action="store_true",
                        help="Force rebuilding sample_inputs.json even if it already exists.")
    parser.add_argument("--show_history_turns", type=int, default=6,
                        help="How many trailing conversation messages to PRINT in the report "
                             "(the model always receives the full history).")
    parser.add_argument("--show_full_history", action="store_true",
                        help="Print the entire conversation in the report instead of the last K messages.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
