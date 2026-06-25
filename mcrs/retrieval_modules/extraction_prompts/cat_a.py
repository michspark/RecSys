"""Category A extraction prompts — 오디오 특성/분위기로 트랙 찾기."""

# A-HL (26세션): 구체적 오디오 특성 명시
PROMPT_HL = """You are extracting audio-focused search keywords from a music conversation.

The user is describing SPECIFIC audio characteristics they want:
instruments, tempo, production style, subgenre details.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: only if the user explicitly names an artist they want MORE of.
  If they say "like Apollo Brown" as a reference, include it.
  If they say "from other artists too", set to null.
- tag_list: the CORE audio descriptors from the LAST message.
  Include: subgenre (boom-bap, liquid D&B, trip-hop),
  instruments (saxophone, synth, piano),
  production style (lo-fi, polished, raw),
  energy (chill, driving, intense),
  texture (atmospheric, dark, bright, warm).
  Prioritize words the user actually used.
- clap_keywords: same as tag_list but add broader sonic descriptors
  that describe how the music SOUNDS, not metadata.
  Example: "mellow beats, warm bass, jazzy samples, laid-back groove"
- rejected: artists or styles the user said NO to.

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "comma-separated audio descriptors",
  "clap_keywords": "comma-separated sonic descriptors for audio matching",
  "rejected": []
}}"""


# A-LH (13세션): 기억 속 특정 곡 찾기
PROMPT_LH = """You are helping find a SPECIFIC track the user is trying to remember.

The user has a song in mind but can't recall the exact name.
They are giving clues about how it SOUNDS.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: any artist the user mentions or confirms.
  If they said "YES! That's the one!" about a previous recommendation,
  include that artist.
- tag_list: combine TWO types of keywords:
  1. Identity clues: artist names, genre, era mentioned anywhere
  2. Sound clues: the sonic description from the LAST message
  Example: "math rock, bouncy guitar, intricate, Chon, clean tone"
- clap_keywords: focus purely on the SOUND description.
  What does the track sound like? How does it feel?
  Example: "bouncy melody, complex rhythm, clean guitar, upbeat energy"
- found: true if the user confirmed they found the track (said "YES",
  "that's it", "perfect", "you found it"), false otherwise.
  If found=true, tag_list should shift to "similar to [found track]" mode.
- rejected: tracks/artists the user said "not quite" or "not the one".

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "identity + sound clues",
  "clap_keywords": "pure sound descriptors",
  "found": false,
  "rejected": []
}}"""


# A-LL (22세션): 넓은 분위기 탐색
PROMPT_LL = """You are extracting mood and atmosphere keywords from a music conversation.

The user is exploring a BROAD sonic mood or atmosphere.
They don't have a specific track or artist in mind.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: only if the user explicitly asks for a specific artist
  or says "I love [artist], more like them".
  For "from a different artist" or general browsing, set to null.
- tag_list: atmospheric and mood keywords that describe the VIBE.
  Include: mood (dark, uplifting, melancholic, haunting),
  atmosphere (immersive, spacious, intimate),
  energy (calm, intense, driving),
  genre only if mentioned (atmospheric metal, ambient, downtempo).
  Keep it broad — don't over-specify.
- clap_keywords: translate the mood into SONIC descriptions.
  How would this music SOUND?
  Example for "dark atmospheric with female vocals":
  "dark atmosphere, heavy bass, ethereal female voice,
   slow tempo, reverb-heavy, immersive soundscape"
- continue_from: if user liked the previous track ("yes!", "perfect!"),
  include the previous artist/genre to stay in that direction.
- rejected: anything user said doesn't fit ("too heavy", "not quite").

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "mood and atmosphere keywords",
  "clap_keywords": "sonic translation of the mood",
  "continue_from": "previous liked artist/genre" or null,
  "rejected": []
}}"""


PROMPTS = {
    "HL": PROMPT_HL,
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "HH": PROMPT_HL,  # A에는 HH 세션이 없으므로 HL로 fallback
    "default": PROMPT_HL,
}


# ── 쿼리 빌더 ─────────────────────────────────────────────────────────────────

def _build_bge_query(data: dict) -> str:
    """BGE 메타데이터 인덱스 검색용 쿼리 문자열 구성."""
    parts = []
    if data.get("artist_name"):
        parts.append(f"artist_name: {data['artist_name']}")
    if data.get("tag_list"):
        parts.append(f"tag_list: {data['tag_list']}")
    # A-LL: continue_from이 있으면 artist_name으로 추가해 BGE 매칭 강화
    if data.get("continue_from"):
        parts.append(f"artist_name: {data['continue_from']}")
    return "\n".join(parts)


def build_result(data: dict, _specificity: str, fallback: str) -> dict:
    """JSON 파싱 결과를 retriever가 쓰는 형식으로 변환."""
    bge_query     = _build_bge_query(data) or fallback
    clap_keywords = data.get("clap_keywords") or data.get("tag_list") or fallback

    result = {
        "direct_request": None,  # Category A에서는 직접 요청이 거의 없음
        "bge_query":      bge_query,
        "clap_keywords":  clap_keywords,
        "rejected":       data.get("rejected") or [],
    }
    # 카테고리 전용 필드 그대로 전달
    for key in ("found", "continue_from"):
        if key in data:
            result[key] = data[key]
    return result
