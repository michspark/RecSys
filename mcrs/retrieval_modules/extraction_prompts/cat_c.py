"""Category C extraction prompts — 앨범 커버/비주얼 기반 트랙 찾기."""

# C-HH: 비주얼 스타일 + 음악 장르 동시에 명시
PROMPT_HH = """You are extracting visual and audio search keywords from a music conversation.

The user is describing BOTH the visual style they remember AND specific musical characteristics.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: if the user mentions or confirms a specific artist.
- visual_keywords: the VISUAL characteristics of the album cover or artwork.
  Include: colors, shapes, photography vs illustration, mood of the image,
  any recognizable visual elements (person, landscape, abstract, typography).
  Example: "dark background, blurry figure, high contrast, monochrome"
- tag_list: the MUSICAL characteristics (genre, style, era, mood).
  Example: "post-punk, dark wave, 80s, atmospheric, melancholic"
- clap_keywords: translate the musical mood into SONIC descriptors.
  Example: "cold synths, reverb-heavy guitar, distant drums, brooding atmosphere"
- rejected: artists, albums, or visual styles the user explicitly rejected.

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "visual_keywords": "visual description of the album artwork",
  "tag_list": "musical genre/style/mood descriptors",
  "clap_keywords": "sonic descriptors",
  "rejected": []
}}"""


# C-HL: 비주얼로 시작하지만 음악 장르도 점차 좁혀가는 패턴
PROMPT_HL = """You are helping find an album the user remembers by its VISUAL STYLE and musical genre.

The user remembers the album cover clearly and is combining visual memory with genre clues.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: if the user confirms or strongly suspects an artist.
  Example: "I think it's Burial or something like that" → artist_name = "Burial"
- visual_keywords: the VISUAL elements they describe.
  Be specific about: color palette, subject matter, art style, composition.
  Example: "grainy urban photo, dark street, neon lights, rain-slicked pavement"
  Use the user's LAST message words as much as possible.
- tag_list: musical genre + any style clues from the LAST message.
  This evolves per turn — extract ONLY from the LAST message.
  Example: "electronic, UK bass, underground, dark, minimal"
- clap_keywords: sonic translation of what the music sounds like.
  Example: "chopped samples, sub-bass, lo-fi textures, urban atmosphere"
- found: true if the user confirmed finding the album ("YES!", "that's it!").
  After found=true, shift focus to similar albums.
- rejected: albums or visual styles that were close but wrong.

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "visual_keywords": "visual description from LAST message",
  "tag_list": "genre and style from LAST message",
  "clap_keywords": "sonic descriptors",
  "found": false,
  "rejected": []
}}"""


# C-LH: 앨범 커버 아트만 기억하는 경우 (가장 흔한 패턴, 32/58 세션)
PROMPT_LH = """You are helping find an album the user remembers ONLY by its cover art.

The user has a clear visual memory of the album cover but doesn't know the artist or title.
They are narrowing down through visual clues across multiple turns.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: any artist the user mentions or confirms.
  If they say "YES! That's the album!", include that artist.
  If still searching, include guesses: "maybe Joy Division or The Cure"
- visual_keywords: the CORE visual description from the user's LAST message.
  This is the PRIMARY search signal — be thorough and specific.
  Include ALL visual details mentioned: colors, objects, people, setting,
  art style (photo vs painting vs illustration), composition, mood of image.
  Example: "white background, simple black drawing, stick figure, minimalist"
  Example: "four men walking across a zebra crossing, daytime, black and white photo"
  Use the user's exact words when possible.
- tag_list: any musical genre or style clues the user mentioned.
  If they said "it's a rock album" earlier in conversation, include "rock".
  Set to null if no musical clues given at all.
- clap_keywords: if the user mentioned anything about the SOUND, translate it.
  Otherwise set to null.
- found: true if the user confirmed the album ("YES!", "that's it!", "you found it!").
- rejected: albums that were visually similar but wrong.
  Include WHY they were rejected: "similar cover but wrong era"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "visual_keywords": "detailed visual description from LAST message",
  "tag_list": "musical genre/style if mentioned" or null,
  "clap_keywords": "sonic descriptors if mentioned" or null,
  "found": false,
  "rejected": []
}}"""


# C-LL: 막연한 비주얼 기억 + 음악 분위기로 탐색
PROMPT_LL = """You are helping find music based on VAGUE visual memories and general mood.

The user has a fuzzy memory of the album cover and is exploring by mood and general vibe.
They don't have a specific album in mind — just fragments of a visual feeling.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: only if the user explicitly names an artist they want.
  For general browsing or vague memories, set to null.
- visual_keywords: even vague visual impressions are useful.
  Example: "dark, moody, probably black and white, felt serious"
  Example: "colorful, abstract, trippy, psychedelic feeling"
  Use whatever visual fragments the user mentions.
- tag_list: the musical mood and genre they associate with the visual memory.
  Example: "dark, atmospheric, introspective, probably indie or alternative"
- clap_keywords: how would this music SOUND based on the visual mood?
  Example: "slow tempo, reverb-heavy, atmospheric textures, melancholic tone"
- mood_shift: describe how the user's request changed from the previous turn.
  "from dark/heavy to lighter/melodic", "from vague to more specific visual"
  null if first turn or no change.
- rejected: albums or moods that didn't fit.

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "visual_keywords": "vague visual impression",
  "tag_list": "musical mood and genre",
  "clap_keywords": "sonic descriptors matching the mood",
  "mood_shift": "..." or null,
  "rejected": []
}}"""


PROMPTS = {
    "HH": PROMPT_HH,
    "HL": PROMPT_HL,
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "default": PROMPT_LH,  # C에서 specificity 미상이면 LH가 가장 흔한 패턴
}


# ── 쿼리 빌더 ─────────────────────────────────────────────────────────────────

def _build_bge_query(data: dict, specificity: str) -> str:
    """BGE 메타데이터 인덱스 검색용 쿼리 문자열 구성."""
    parts = []
    if data.get("artist_name"):
        parts.append(f"artist_name: {data['artist_name']}")
    if data.get("tag_list"):
        parts.append(f"tag_list: {data['tag_list']}")
    return "\n".join(parts)


def _build_attr_query(data: dict, specificity: str) -> str:
    """attributes-qwen3 임베딩 검색용 쿼리 구성."""
    parts = []
    if data.get("tag_list"):
        parts.append(data["tag_list"])
    if data.get("clap_keywords"):
        parts.append(data["clap_keywords"])
    return ", ".join(parts)


def _build_image_query(data: dict, specificity: str) -> str:
    """SigLIP2 비주얼 임베딩 검색용 텍스트 쿼리 구성.

    visual_keywords를 SigLIP2의 텍스트 인코더에 넣을 자연어 설명으로 변환.
    """
    if data.get("visual_keywords"):
        return data["visual_keywords"]
    # visual_keywords 없으면 tag_list로 fallback
    return data.get("tag_list") or ""


def build_result(data: dict, specificity: str, fallback: str) -> dict:
    """JSON 파싱 결과를 retriever가 쓰는 형식으로 변환."""
    bge_query   = _build_bge_query(data, specificity) or fallback
    attr_query  = _build_attr_query(data, specificity) or fallback
    image_query = _build_image_query(data, specificity) or fallback

    clap_keywords = data.get("clap_keywords") or data.get("tag_list") or fallback

    result = {
        "direct_request":  data.get("direct_request"),
        "bge_query":       bge_query,
        "clap_keywords":   clap_keywords,
        "attr_query":      attr_query,   # attributes-qwen3 검색용
        "image_query":     image_query,  # SigLIP2 텍스트 쿼리
        "visual_keywords": data.get("visual_keywords"),
        "rejected":        data.get("rejected") or [],
    }
    # 카테고리 전용 필드 그대로 전달
    for key in ("found", "mood_shift"):
        if key in data:
            result[key] = data[key]
    return result
