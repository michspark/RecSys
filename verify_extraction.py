#!/usr/bin/env python3
"""
verify_extraction.py — cat_a ~ cat_k 전체 44가지 키워드 추출 검증 스크립트

사용법:
  python verify_extraction.py --model Qwen/Qwen3-4B          # 전체 44케이스 실행
  python verify_extraction.py --model Qwen/Qwen3-4B --compact # 한 줄 로그 형식
  python verify_extraction.py --model Qwen/Qwen3-4B --cat G H # G,H만
  python verify_extraction.py --model Qwen/Qwen3-4B --spec LH # LH만 (11케이스)
  python verify_extraction.py --model Qwen/Qwen3-4B --logfile extraction.log  # 타임스탬프로 append
  python verify_extraction.py                                 # 모델 없이 프롬프트만 확인
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── 모듈 직접 로드 (bm25s 등 무거운 의존성 우회) ─────────────────────────────

PROMPTS_DIR = Path(__file__).parent / "mcrs" / "retrieval_modules" / "extraction_prompts"

def _load_module(name: str):
    """extraction_prompts 안의 cat_*.py를 패키지 import 없이 로드."""
    path = PROMPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

CAT_MODULES = {
    cat: _load_module(f"cat_{cat.lower()}")
    for cat in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
}

ALL_SPECS = ["HH", "HL", "LH", "LL"]

# ── 샘플 대화 — 44가지 (11 cats × 4 specs) ───────────────────────────────────

SAMPLES: dict[str, dict[str, str]] = {
    "A": {
        "HH": (
            "user: I need something instrumental with heavy boom-bap beats, "
            "deep sub-bass, lo-fi vinyl texture, around 80-90 BPM"
        ),
        "HL": (
            "user: I want instrumental hip-hop\n"
            "assistant: Here's something laid-back.\n"
            "user: I like it but I want more boom-bap feel, "
            "deeper bass, maybe some jazzy samples, still lo-fi"
        ),
        "LH": (
            "user: There's this song I can't remember — it had a really bouncy guitar, "
            "super intricate rhythm, almost like math rock but lighter\n"
            "assistant: Could it be something like Chon?\n"
            "user: Yes! Chon sounds exactly right! That clean guitar tone was it"
        ),
        "LL": (
            "user: I want something dark and atmospheric tonight\n"
            "assistant: Here's a dark ambient track.\n"
            "user: Perfect! Can we keep that vibe but add some female vocals?"
        ),
    },
    "B": {
        "HH": (
            "user: I'm looking for a song with the line "
            "'do you have the time to listen to me whine' — I think it's Green Day?\n"
            "assistant: That's 'Basket Case' by Green Day!\n"
            "user: Yes! Can you play another Green Day song with that same punk energy?"
        ),
        "HL": (
            "user: I love Anitta, can you play something that shows her female empowerment side?\n"
            "assistant: Here's 'Vai Malandra' by Anitta.\n"
            "user: Nice! Now I want something more upbeat and dance-focused from her"
        ),
        "LH": (
            "user: I'm looking for a hip-hop song — it's a vivid personal story, "
            "like watching a movie, very specific details about a real event in the rapper's life\n"
            "assistant: Could be something like DUCKWORTH by Kendrick Lamar?\n"
            "user: YES! DUCKWORTH by Kendrick, that's the one!"
        ),
        "LL": (
            "user: I like Weezer, show me some of their songs with interesting lyrics\n"
            "assistant: Here's 'El Scorcho' by Weezer.\n"
            "user: Fun! Now I want something more melancholic and introspective from Weezer"
        ),
    },
    "C": {
        "HH": (
            "user: I'm looking for a post-punk album from the 80s — the cover is all black "
            "with a blurry figure, very monochrome and dark. The music is cold synths and reverb"
        ),
        "HL": (
            "user: I remember an album with a grainy urban photo on the cover, "
            "dark street with neon lights, very UK underground feel\n"
            "assistant: Could this be Burial - Untrue?\n"
            "user: Close! But the cover had more rain-slicked pavement, and it was more minimal electronic"
        ),
        "LH": (
            "user: I'm trying to find an album but only remember the cover — "
            "there were four men walking across a zebra crossing, "
            "black and white photo, daytime\n"
            "assistant: That sounds like Abbey Road by The Beatles!\n"
            "user: YES! Abbey Road, that's exactly it!"
        ),
        "LL": (
            "user: I remember an album with a really colorful, trippy cover — "
            "maybe abstract shapes? Felt psychedelic and experimental\n"
            "assistant: Something like Tame Impala's Currents?\n"
            "user: Maybe, I want something like that but with a heavier sound"
        ),
    },
    "D": {
        "HH": (
            "user: Play 'Bohemian Rhapsody' by Queen\n"
            "assistant: Here's Bohemian Rhapsody by Queen.\n"
            "user: Perfect! Do you have other Queen songs with big operatic sections?"
        ),
        "HL": (
            "user: I want dark contemplative jazz for late-night studying — "
            "quiet piano, smoky atmosphere, very noir\n"
            "assistant: Here's some Bill Evans.\n"
            "user: Good but I want even more minimal, just solo piano, slower tempo"
        ),
        "LH": (
            "user: I'm looking for that epic track from Moana — "
            "the one that feels like sailing across the ocean for the first time, "
            "very grand and adventurous\n"
            "assistant: Is it 'How Far I'll Go' or 'Shiny'?\n"
            "user: Neither, it was more orchestral and instrumental, "
            "building to a huge climax, sense of destiny"
        ),
        "LL": (
            "user: Play me some Red Hot Chili Peppers, something chill\n"
            "assistant: Here's 'Soul to Squeeze'.\n"
            "user: Nice! Now give me something more funky and upbeat from them, "
            "heavy on the bass"
        ),
    },
    "E": {
        "HH": (
            "user: I'm looking for tracks in D# minor, around 118 BPM, "
            "experimental electronic — similar to Aphex Twin's Drukqs album"
        ),
        "HL": (
            "user: Take me on a journey from classic EBM to modern synthpop\n"
            "assistant: Starting with Front 242.\n"
            "user: Good start! Now move toward something more melodic, "
            "futurepop territory, keep it dark but add melody"
        ),
        "LH": (
            "user: I love 'Take Five' by Dave Brubeck but I don't know why exactly\n"
            "assistant: Is it the cool jazz vibe or the saxophone?\n"
            "user: Hmm, maybe the rhythm? It feels unusual somehow\n"
            "assistant: It's in 5/4 time — very uncommon!\n"
            "user: YES! That's exactly it! The weird time signature and how the melody "
            "flows over it — I want more songs with unusual time signatures"
        ),
        "LL": (
            "user: I want to discover something completely new — fresh and underground\n"
            "assistant: Here's some experimental hip-hop.\n"
            "user: Interesting! I liked the electronic textures, "
            "can you go more atmospheric? Less beats, more immersive"
        ),
    },
    "F": {
        "HH": (
            "user: Play 'Numb' by Linkin Park from Hybrid Theory\n"
            "assistant: Here's Numb by Linkin Park.\n"
            "user: Great! What other tracks are on Hybrid Theory? "
            "I want the heavier ones"
        ),
        "HL": (
            "user: I want late 90s electronic dance tracks, energetic, for a party\n"
            "assistant: Here's some late 90s trance.\n"
            "user: Good era! But I want something deeper — "
            "more underground house, heavy bassline, atmospheric, still late 90s"
        ),
        "LH": (
            "user: I'm trying to find a 70s jazz-funk track — "
            "it had a prominent vibraphone and a really groovy metropolitan feel\n"
            "assistant: Could be Roy Ayers or Donald Byrd?\n"
            "user: It might be Roy Ayers! The track was very laid-back, "
            "vibraphone was the main instrument, very smooth"
        ),
        "LL": (
            "user: I want to explore classic jazz — just the good stuff\n"
            "assistant: Here's Miles Davis.\n"
            "user: Miles Davis is always great! "
            "Now I want to hear more piano-focused jazz, exceptional solos"
        ),
    },
    "G": {
        "HH": (
            "user: Play the piano version of 'Particles' by Nothing But Thieves\n"
            "assistant: Here's the stripped-back piano version.\n"
            "user: That was beautiful — raw and melancholic. "
            "Can you find more acoustic or piano covers of rock songs with that emotional vulnerability?"
        ),
        "HL": (
            "user: I want songs that express profound sadness but with resilient determination — "
            "like crying but still standing strong\n"
            "assistant: Here's 'Rise Up' by Andra Day.\n"
            "user: Yes that direction! But I want something more dramatic, "
            "sweeping orchestral feel, powerful female vocals building to a climax"
        ),
        "LH": (
            "user: I'm trying to find a song — it gave me this desolate winter feeling, "
            "very haunting, slow and atmospheric\n"
            "assistant: Could be something like Lycia or Switchblade Symphony?\n"
            "user: Something like that yes! It had ethereal female vocals, "
            "very darkwave, cold and beautiful, almost gothic"
        ),
        "LL": (
            "user: I need something to cheer me up — positive and uplifting\n"
            "assistant: Here's 'Joy' by for KING & COUNTRY.\n"
            "user: Perfect vibe! Can you give me more like that — "
            "encouraging, about hope and perseverance"
        ),
    },
    "H": {
        "HH": (
            "user: Play 'White & Nerdy' by Weird Al Yankovic\n"
            "assistant: Here's White & Nerdy by Weird Al.\n"
            "user: Haha perfect! Play another Weird Al song — "
            "something equally funny and catchy"
        ),
        "HL": (
            "user: I want to explore Green Day's discography — "
            "start with their early punk sound from the 90s\n"
            "assistant: Here's 'Longview' from Dookie.\n"
            "user: Great early sound! Now I want something from their more mature era, "
            "American Idiot period, more political and anthemic"
        ),
        "LH": (
            "user: There's an electronic band I can't remember — "
            "they had this powerful, almost anthemic sound, "
            "very epic and emotional, driving beat\n"
            "assistant: Could be VNV Nation or Assemblage 23?\n"
            "user: YES! VNV Nation, that's exactly the sound! "
            "That epic introspective electronic with emotional male vocals"
        ),
        "LL": (
            "user: I'm in the mood to explore new artists — surprise me\n"
            "assistant: Here's some K-Pop from EXID.\n"
            "user: I like this! Very energetic. "
            "Give me more K-Pop girl groups from around 2013-2016, "
            "that powerful punchy sound"
        ),
    },
    "I": {
        "HH": (
            "user: Play 'Africa' by Toto\n"
            "assistant: Here's Africa by Toto.\n"
            "user: I love that African-inspired 80s sound! "
            "Can you find more tracks with that exotic continental feel?"
        ),
        "HL": (
            "user: I want to explore bossa nova — Brazilian jazz from the 60s\n"
            "assistant: Here's João Gilberto.\n"
            "user: Beautiful! Very sophisticated. "
            "Now I want something more upbeat bossa nova, more samba influence"
        ),
        "LH": (
            "user: I'm looking for a non-English song — it was a huge hit "
            "around 2015-2017, very danceable, Latin pop feel, female vocalist\n"
            "assistant: Could be Shakira or J Balvin?\n"
            "user: I think it was a Spanish female artist, "
            "very catchy chorus, tropical beat"
        ),
        "LL": (
            "user: I want to discover music from different cultures — "
            "something I wouldn't normally hear\n"
            "assistant: Here's some K-Pop.\n"
            "user: Fun! Now something completely different — "
            "maybe African or Latin music, high energy, rhythmic"
        ),
    },
    "J": {
        "HH": (
            "user: Play 'Smells Like Teen Spirit' by Nirvana\n"
            "assistant: Here's Smells Like Teen Spirit by Nirvana.\n"
            "user: Iconic! What other grunge tracks from the early 90s "
            "were equally defining and popular?"
        ),
        "HL": (
            "user: I want tracks that are beloved by outlaw country fans — "
            "the gritty storytelling kind\n"
            "assistant: Here's Waylon Jennings.\n"
            "user: Perfect! That raw authenticity is what I'm after. "
            "More outlaw country classics with rebellious spirit"
        ),
        "LH": (
            "user: I'm trying to remember a huge hip-hop hit from the early 90s — "
            "it was everywhere, everyone knew it, East Coast style, very gritty\n"
            "assistant: Could be Notorious B.I.G. or Nas?\n"
            "user: It was more of a boom-bap beat, hard-hitting, "
            "New York underground but still a massive hit"
        ),
        "LL": (
            "user: Just give me the biggest hits from the 90s — "
            "I want nostalgia, party songs everyone knows\n"
            "assistant: Here's 'Wannabe' by Spice Girls.\n"
            "user: Yes! More 90s anthems like that — "
            "the kind that everyone sang along to at parties"
        ),
    },
    "K": {
        "HH": (
            "user: What was the defining East Coast hip-hop track from 1995? "
            "Something that captured the whole era"
        ),
        "HL": (
            "user: I want to explore The Black Keys' catalog — "
            "start with their gritty blues-rock core sound\n"
            "assistant: Here's 'Howlin' For You'.\n"
            "user: That raw guitar sound is perfect! "
            "Now give me something more late-night and melancholic from them, "
            "same bluesy vibe but slower"
        ),
        "LH": (
            "user: I'm trying to find a 90s hip-hop track — "
            "everyone knew it, very New York, grimy\n"
            "assistant: Mobb Deep or Wu-Tang?\n"
            "user: More Wu-Tang style — that dark cinematic boom-bap, "
            "sample-heavy production, East Coast street feel\n"
            "assistant: 'C.R.E.A.M.' or 'Protect Ya Neck'?\n"
            "user: YES! C.R.E.A.M. that's the sound I was thinking of"
        ),
        "LL": (
            "user: I just want to discover great music — no constraints, surprise me\n"
            "assistant: Here's some Philip Glass.\n"
            "user: Interesting! I liked the minimalist feel. "
            "More contemporary classical or ambient, something atmospheric and melancholic"
        ),
    },
}

# ── 카테고리별 핵심 체크 필드 ─────────────────────────────────────────────────
# compact 모드에서 어떤 필드가 나왔는지 확인

KEY_FIELDS: dict[str, list[str]] = {
    "A": ["artist_name", "tag_list", "clap_keywords", "found", "continue_from"],
    "B": ["artist_name", "lyric_keywords", "tag_list", "phase", "found"],
    "C": ["artist_name", "visual_keywords", "tag_list", "found", "mood_shift"],
    "D": ["artist_name", "source", "tag_list", "found", "frustrated"],
    "E": ["artist_name", "tag_list", "discovery_phase", "journey_stage", "discovered_elements"],
    "F": ["artist_name", "tag_list", "found", "instrument_focus", "era_locked"],
    "G": ["artist_name", "tag_list", "emotional_core", "found", "mood_label"],
    "H": ["artist_name", "tag_list", "same_artist_mode", "found", "locked_direction"],
    "I": ["artist_name", "tag_list", "culture_region", "found"],
    "J": ["artist_name", "tag_list", "niche_community", "popularity_era", "found"],
    "K": ["artist_name", "tag_list", "target_year", "era_anchor", "browse_mode"],
}

# ── 출력 포맷 헬퍼 ─────────────────────────────────────────────────────────────

WIDTH = 70

def _hr(char: str = "═") -> str:
    return char * WIDTH

def _header(text: str) -> str:
    pad = (WIDTH - len(text) - 2) // 2
    return f"{'═' * pad} {text} {'═' * (WIDTH - pad - len(text) - 2)}"

def _section(title: str, content: str) -> str:
    lines = [f"  ┌─ {title}", "  │"]
    for line in content.strip().splitlines():
        lines.append(f"  │  {line}")
    lines.append("  │")
    return "\n".join(lines)

def _json_pretty(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=4)

def _compact_line(cat: str, spec: str, result: dict, issues: list[str]) -> str:
    """단일 케이스를 한 줄 로그로 요약."""
    bge  = result.get("bge_query", "")[:50].replace("\n", " | ")
    clap = result.get("clap_keywords", "")
    if isinstance(clap, list):
        clap = ", ".join(clap)
    clap = str(clap)[:40]

    # 카테고리별 핵심 필드만 추출
    extras = []
    for f in KEY_FIELDS.get(cat, []):
        val = result.get(f)
        if val is not None and val is not False and val != []:
            extras.append(f"{f}={val!r}")
    extra_str = "  " + " | ".join(extras) if extras else ""

    issue_str = "  ⚠ " + "; ".join(issues) if issues else ""
    status = "✓" if not issues else "⚠"

    return f"  [{status}] {cat}-{spec}  bge='{bge}'  clap='{clap}'{extra_str}{issue_str}"

# ── Qwen3 추출 로직 ───────────────────────────────────────────────────────────

def _last_user_message(conversation_str: str) -> str:
    last = ""
    for line in conversation_str.strip().split("\n"):
        line = line.strip()
        if line.startswith("user:"):
            last = line[5:].strip()
    return last if last else conversation_str[:200]

_TEXT_FIELDS = {
    "tag_list", "clap_keywords", "lyric_keywords",
    "visual_keywords", "discovered_elements", "constraints",
}

def _normalize_data(data: dict) -> tuple[dict, list[str]]:
    """list→str 변환 + 이슈 감지."""
    issues = []
    for field in _TEXT_FIELDS:
        if field in data and isinstance(data[field], list):
            issues.append(f"{field} was list→joined")
            data[field] = ", ".join(str(v) for v in data[field])
    return data, issues

def _run_qwen(system_prompt: str, model, tokenizer, device: str) -> str:
    import torch
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": ""},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        raw = m.group() if m else raw
    return raw.strip()

# ── 단일 케이스 실행 ──────────────────────────────────────────────────────────

def run_one(
    cat: str,
    spec: str,
    conversation: str,
    lm_components: Optional[tuple],
    output_lines: list[str],
    compact: bool = False,
) -> None:
    mod      = CAT_MODULES[cat]
    last_msg = _last_user_message(conversation)

    prompt_template = mod.PROMPTS.get(spec, mod.PROMPTS.get("default", ""))
    system_prompt = (
        prompt_template.format(last_user_msg=last_msg, conversation_str=conversation)
        if "{last_user_msg}" in prompt_template
        else prompt_template
    )

    if not compact:
        output_lines.append(_hr())
        output_lines.append(_header(f"CATEGORY {cat}  |  SPECIFICITY {spec}"))
        output_lines.append(_hr())
        output_lines.append(_section("SAMPLE CONVERSATION", conversation))
        prompt_preview = "\n".join(system_prompt.splitlines()[:50])
        if len(system_prompt.splitlines()) > 50:
            prompt_preview += f"\n  ... ({len(system_prompt.splitlines()) - 50} more lines)"
        output_lines.append(_section("PROMPT TEMPLATE (first 50 lines)", prompt_preview))

    if lm_components is None:
        if compact:
            output_lines.append(f"  [–] {cat}-{spec}  (no model)")
        else:
            output_lines.append("  ⚠  No model — skipping inference.")
            output_lines.append("")
        return

    model, tokenizer, device = lm_components
    try:
        raw = _run_qwen(system_prompt, model, tokenizer, device)

        if not compact:
            output_lines.append(_section("RAW QWEN3 OUTPUT", raw))

        data = json.loads(raw)
        data, issues = _normalize_data(data)

        if not compact:
            output_lines.append(_section("PARSED JSON", _json_pretty(data)))

        builder = getattr(mod, "build_result", None)
        result  = builder(data, spec, last_msg) if builder else {}

        if compact:
            output_lines.append(_compact_line(cat, spec, result, issues))
        else:
            if builder:
                output_lines.append(_section("BUILD RESULT", _json_pretty(result)))
            # 이슈가 있으면 verbose 모드에도 표시
            if issues:
                output_lines.append(f"  ⚠  Issues: {'; '.join(issues)}")
            output_lines.append("")

    except json.JSONDecodeError as exc:
        raw_preview = repr(raw[:80])
        msg = f"JSON parse error: {exc} | raw={raw_preview}"
        if compact:
            output_lines.append(f"  [✗] {cat}-{spec}  {msg}")
        else:
            output_lines.append(f"  ✗  {msg}")
            output_lines.append("")
    except Exception as exc:
        msg = f"Error: {exc}"
        if compact:
            output_lines.append(f"  [✗] {cat}-{spec}  {msg}")
        else:
            output_lines.append(f"  ✗  {msg}")
            output_lines.append("")

# ── 모델 로딩 ─────────────────────────────────────────────────────────────────

def load_model(model_name: str) -> tuple:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify] Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device).eval()
    print(f"[verify] Model ready.\n")
    return model, tokenizer, device

# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="cat_a ~ cat_k 키워드 추출 검증 (44케이스)")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace 모델 이름 (예: Qwen/Qwen3-4B)")
    parser.add_argument("--cat",  nargs="+", choices=list(CAT_MODULES.keys()),
                        default=list(CAT_MODULES.keys()),
                        help="검증할 카테고리 (예: --cat A B G)")
    parser.add_argument("--spec", nargs="+", choices=ALL_SPECS,
                        default=ALL_SPECS,
                        help="검증할 specificity (예: --spec LH HL)")
    parser.add_argument("--outfile", type=str, default=None,
                        help="결과를 파일로 저장 (덮어쓰기)")
    parser.add_argument("--logfile", type=str, default=None,
                        help="타임스탬프로 append하는 로그 파일 (예: extraction.log)")
    parser.add_argument("--compact", action="store_true",
                        help="한 줄 로그 형식으로 출력 (빠른 전체 확인용)")
    args = parser.parse_args()

    lm_components = load_model(args.model) if args.model else None

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_label = args.model if args.model else "None (prompt-only)"
    n_cases = len(args.cat) * len(args.spec)

    output_lines: list[str] = []

    if args.compact:
        output_lines.append(f"\n── {ts}  model={model_label}  {n_cases} cases ──")
    else:
        output_lines.append(f"\n{'═' * WIDTH}")
        output_lines.append(f"  EXTRACTION VERIFICATION  |  {ts}")
        output_lines.append(f"  cats={args.cat}  specs={args.spec}  ({n_cases} cases)")
        output_lines.append(f"  model={model_label}")
        output_lines.append(f"{'═' * WIDTH}\n")

    ok = err = warned = 0
    for cat in args.cat:
        for spec in args.spec:
            conversation = SAMPLES[cat][spec]
            before = len(output_lines)
            run_one(cat, spec, conversation, lm_components, output_lines, compact=args.compact)
            # 결과 줄에서 상태 파악
            new_lines = output_lines[before:]
            if any("✗" in l for l in new_lines):
                err += 1
            elif any("⚠" in l for l in new_lines):
                warned += 1
            elif lm_components is not None:
                ok += 1

    if args.compact:
        output_lines.append(f"── done: ✓{ok}  ⚠{warned}  ✗{err} ──\n")
    else:
        output_lines.append(_hr("─"))
        output_lines.append(f"  Done: ✓{ok} ok  ⚠{warned} warnings  ✗{err} errors  "
                            f"({n_cases} total)")
        output_lines.append(_hr("─"))

    result_text = "\n".join(output_lines)
    print(result_text)

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(result_text)
        print(f"[verify] Saved to {args.outfile}")

    if args.logfile:
        with open(args.logfile, "a", encoding="utf-8") as f:
            f.write(result_text + "\n")
        print(f"[verify] Appended to {args.logfile}")

if __name__ == "__main__":
    main()
