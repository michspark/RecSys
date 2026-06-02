"""Category G extraction prompts — 감정/무드 기반 트랙 찾기."""

# G-HH: 특정 버전/리믹스/에디트 찾기
PROMPT_HH = """You are extracting exact track version identification from a music conversation.

The user is looking for a SPECIFIC VERSION, REMIX, or EDIT of a song,
or a specific cover that changes the original's mood/style.
After finding it, they explore similar heavy/intense music.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- direct_request: if LAST message asks for a specific track+version.
  Include track_name, artist_name, and version_info.
  Example: "Play the Cervical Edit of Pantera's Walk"
  → {{"track_name": "Walk", "artist_name": "Pantera", "version": "Cervical Edit"}}
  Example: "Play the piano version of 'Particles'"
  → {{"track_name": "Particles", "artist_name": "Nothing But Thieves", "version": "piano version"}}
  Set to null if user is browsing, not requesting a specific version.
- artist_name: current artist being explored.
- tag_list: genre and style descriptors.
  After finding the specific version, user often wants more of that SOUND:
  "heavy groove metal, crushing riffs, slow breakdowns"
  "melancholic piano, acoustic, stripped-back, raw emotion"
  Extract from the LAST message.
- mood_quality: the emotional quality the user is chasing.
  "aggressive, head-caving, brutal" or "melancholic, raw, vulnerable"
  This drives the mood-based follow-up recommendations.
- wants_similar_mood: true if user found their track and wants more
  with the same emotional impact. false if still searching.
- rejected: versions or tracks that weren't right.

Output ONLY valid JSON:
{{
  "direct_request": {{"track_name": "...", "artist_name": "...", "version": "..."}} or null,
  "artist_name": "...",
  "tag_list": "genre, style descriptors",
  "mood_quality": "emotional quality being sought",
  "wants_similar_mood": false,
  "rejected": []
}}"""


# G-HL (31세션): 특정 감정의 여러 곡 탐색
PROMPT_HL = """You are extracting complex emotional and mood keywords from a music conversation.

The user wants multiple tracks that match a SPECIFIC, NUANCED emotional state.
They describe complex feelings — often a MIX of emotions — and refine each turn.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: only if user asks for a specific artist.
  For mood-based browsing, usually null.
  If user says "from other artists" or "not just [artist]", set to null.
- tag_list: the EMOTIONAL LANDSCAPE from the LAST message.
  Capture the COMPLEXITY — users often want CONTRADICTORY moods:
  "sad BUT with strength", "uplifting BUT melancholic",
  "calm BUT with emotional depth", "intense BUT intimate"
  Include: primary emotion, secondary emotion, genre, setting.
  Example: "profound sadness, resilient determination, dramatic,
  powerful female vocals, Broadway, emotional arc, sweeping ballad"
- clap_keywords: translate the emotions into SONIC characteristics.
  How does this emotional state SOUND?
  "profound sadness + strength" → "powerful vocals building to climax,
  orchestral swells, minor key with resolving moments, dramatic dynamics"
  "calm + soulful depth" → "warm bass, gentle Rhodes piano,
  smooth vocals, slow groove, intimate recording, slight reverb"
- emotional_core: the PRIMARY emotional request in 2-3 words.
  "sad but strong" | "calm introspection" | "cathartic anger" |
  "warm intimacy" | "uplifting hope" | "dark contemplation"
- refinement: how the emotion shifted from the previous turn.
  "wants more powerful climax", "less upbeat, more depth",
  "female vocalist specifically", "more acoustic, less electronic"
  Extract from LAST message contrasts with what was recommended.
- rejected: tracks that missed the emotional mark.
  Include the emotional mismatch:
  "too upbeat, missing the sadness", "right mood but not powerful enough",
  "too laid-back, need more emotional intensity"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "complex emotional landscape + genre + style",
  "clap_keywords": "sonic translation of emotional state",
  "emotional_core": "2-3 word emotion summary",
  "refinement": "emotional shift from last turn" or null,
  "rejected": []
}}"""


# G-LH (22세션): 기억 속 감정적 곡 찾기
PROMPT_LH = """You are helping find a specific song the user remembers by its EMOTIONAL IMPACT.

The user has a song in mind but remembers HOW IT MADE THEM FEEL,
not the exact title or lyrics. They narrow down through emotional clues.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: any artist confirmed or guessed.
  If user says "YES! Lycia!" → include confirmed artist.
  If still searching, include guesses: "maybe Switchblade Symphony"
- tag_list: combine emotional clues with any genre/style clues.
  Build CUMULATIVELY from all turns, dropping rejected elements:
  Turn 1: "dark, atmospheric, melancholic, introspective"
  Turn 3: "dark, atmospheric, melancholic, ethereal female vocals"
  Turn 5: "dark, atmospheric, melancholic, ethereal female vocals,
  darkwave, gothic rock"
  Turn 7: "darkwave, gothic, industrial edge, driving beat,
  female vocals, Switchblade Symphony"
- clap_keywords: the SONIC FEELING the user describes.
  Focus on atmosphere and emotional texture:
  "desolate winter feeling, haunting atmosphere, ethereal voice,
  slow dark ambiance, reverb-heavy, cold and beautiful"
- lyric_keywords: if user mentions lyrical themes or content.
  "about loneliness", "spiritual comfort", "hope and faith"
  Set to null if user only describes the feeling, not content.
- found: true if user confirmed finding the song.
  After found, emotional clues should shift to "more like this feeling".
- emotional_memory: the user's CORE emotional memory of the song.
  "desolate winter feeling", "deep sense of comfort and hope",
  "raw cathartic intensity about relationships"
  This stays constant — it's WHAT they're trying to recreate.
- genre_narrowing: how the genre has narrowed across turns.
  "started broad (atmospheric) → now specific (darkwave + industrial)"
  Helps track the search progression.
- rejected: tracks that had the wrong emotional feel.
  Include the emotional mismatch: "too subdued, need more intensity",
  "right darkness but wrong genre", "good but too modern"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "cumulative emotional + genre clues",
  "clap_keywords": "sonic feeling description",
  "lyric_keywords": "lyrical themes" or null,
  "found": false,
  "emotional_memory": "core emotional memory of the song",
  "genre_narrowing": "how genre narrowed across turns" or null,
  "rejected": []
}}"""


# G-LL (19세션): 넓은 감정 요청
PROMPT_LL = """You are extracting simple mood keywords for casual emotional music browsing.

The user wants music that matches a SIMPLE, CLEAR emotional need.
Not a complex emotional landscape — just a straightforward mood request.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: only if user mentions or reacts positively to an artist.
  For general mood browsing, usually null.
  If user says "more like for KING & COUNTRY", include it.
- tag_list: the SIMPLE mood + genre from the LAST message.
  Keep it straightforward — match the user's simplicity:
  "uplifting, positive, encouraging, Christian rock"
  "feel-good, energetic, happy, hip-hop"
  "calm, peaceful, relaxing, acoustic"
  Don't over-complicate — the user's request IS simple.
- clap_keywords: how this simple mood SOUNDS.
  "uplifting" → "bright major key, driving rhythm, soaring vocals,
  positive energy, anthemic chorus"
  "calm" → "gentle acoustic guitar, soft vocals, slow tempo,
  warm harmonics, peaceful atmosphere"
- lyric_keywords: lyrical themes if user mentions them.
  "about hope", "message of perseverance", "about faith",
  "feel-good message"
  Set to null if user only asks for a mood, not lyrical content.
- mood_label: ONE word that captures the mood.
  "uplifting" | "calming" | "energizing" | "comforting" |
  "empowering" | "cheerful" | "peaceful" | "inspiring"
- frustration_level: how frustrated the user is with recommendations.
  "satisfied": recommendations are hitting the mark
  "mild": some misses but still engaged
  "high": user is repeating the same request, system keeps missing
  When high, the LAST message's correction is MOST important.
- rejected: what missed the mood.
  "too thoughtful, need more energy", "right genre but too heavy",
  "not uplifting enough"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "simple mood + genre descriptors",
  "clap_keywords": "sonic translation of mood",
  "lyric_keywords": "lyrical themes" or null,
  "mood_label": "one-word mood summary",
  "frustration_level": "satisfied",
  "rejected": []
}}"""


PROMPTS = {
    "HH": PROMPT_HH,
    "HL": PROMPT_HL,
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "default": PROMPT_HL,  # G에서 specificity 미상이면 HL (31세션으로 최다)
}


# ── 쿼리 빌더 ─────────────────────────────────────────────────────────────────

def _build_bge_query(data: dict) -> str:
    """BGE 메타데이터 인덱스 검색용 쿼리 문자열 구성."""
    parts = []
    if data.get("artist_name"):
        parts.append(f"artist_name: {data['artist_name']}")
    if data.get("tag_list"):
        parts.append(f"tag_list: {data['tag_list']}")
    return "\n".join(parts)


def _build_lyrics_query(data: dict, specificity: str) -> str:
    """lyrics-qwen3 임베딩 검색용 쿼리.

    G-LH와 G-LL에서만 lyric_keywords가 있을 수 있음.
    """
    if specificity in ("LH", "LL") and data.get("lyric_keywords"):
        return data["lyric_keywords"]
    return ""


def build_result(data: dict, specificity: str, fallback: str) -> dict:
    """JSON 파싱 결과를 retriever가 쓰는 형식으로 변환."""
    bge_query     = _build_bge_query(data) or fallback
    clap_keywords = data.get("clap_keywords") or data.get("tag_list") or fallback
    attr_query    = data.get("tag_list") or fallback
    lyrics_query  = _build_lyrics_query(data, specificity) or None

    result = {
        "direct_request": data.get("direct_request"),
        "bge_query":      bge_query,
        "clap_keywords":  clap_keywords,
        "attr_query":     attr_query,
        "lyrics_query":   lyrics_query,
        "rejected":       data.get("rejected") or [],
    }
    # 카테고리 전용 필드 그대로 전달
    for key in ("found", "emotional_memory", "genre_narrowing",
                "emotional_core", "refinement", "mood_quality",
                "wants_similar_mood", "mood_label", "frustration_level",
                "lyric_keywords"):
        if key in data:
            result[key] = data[key]
    return result
