"""Category D extraction prompts — 정확한 곡 요청 / 분위기 탐색 / OST 찾기."""

# D-HH: 정확한 곡/아티스트 요청
PROMPT_HH = """You are extracting exact track identification keywords from a music conversation.

The user is requesting specific songs by title and/or artist,
or exploring a specific artist's catalog after finding what they wanted.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- direct_request: if the LAST message asks to play a specific song,
  include the exact track_name and artist_name.
  Triggers: "play", "put on", "can you find", followed by a song title.
  Example: "Play 'Oração' by A Banda Mais Bonita da Cidade"
  → {{"track_name": "Oração", "artist_name": "A Banda Mais Bonita da Cidade"}}
  If user is NOT requesting a specific song, set to null.
- artist_name: the artist being explored in the LAST message.
  After finding a song, user often asks "more by [artist]" or "similar to [artist]".
  If user asks for a DIFFERENT artist by name, switch to that artist.
  Example: "Can you play 'Numb' by Linkin Park?" → "Linkin Park"
  Example: "Do you have anything by Green Day?" → "Green Day"
- tag_list: genre and style descriptors for finding similar music.
  Example: "rock, alternative rock, energetic, punk-rock"
  Only extract from the LAST message.
- frustrated: true if the user is getting wrong results and repeating
  their request ("This is NOT what I asked for", "I said rock, not sertanejo").
  When frustrated, the corrected request is MOST important.
- rejected: songs or artists that were played but user didn't want.
  Include the correction: "not sertanejo, I want rock"

Output ONLY valid JSON:
{{
  "direct_request": {{"track_name": "...", "artist_name": "..."}} or null,
  "artist_name": "current artist focus",
  "tag_list": "genre, style descriptors",
  "frustrated": false,
  "rejected": []
}}"""


# D-HL: 특정 분위기의 여러 곡 탐색
PROMPT_HL = """You are extracting mood and atmosphere keywords for a themed music session.

The user wants multiple tracks that fit a SPECIFIC mood, setting, or thematic context.
They describe a detailed atmosphere and refine it each turn.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: only if user asks for a specific artist or says "more by [artist]".
  For mood-based browsing ("more dark jazz"), set to null.
  If user says "different artist", set to null.
- tag_list: the ATMOSPHERE and SETTING keywords from the LAST message.
  Include ALL of these if mentioned:
  Setting: "late-night", "driving", "rainy day", "party", "meditation"
  Mood: "dark", "contemplative", "mysterious", "gritty", "raw"
  Genre: "jazz", "folk", "blues", "electronic", "ambient"
  Style: "acoustic", "noir", "cinematic", "atmospheric"
  Era: "70s", "classic", "contemporary"
  Example: "dark, contemplative, noir jazz, acoustic, mysterious,
  shadowy atmosphere, late-night"
- clap_keywords: translate the atmosphere into SONIC descriptions.
  How would this music SOUND?
  Example for "dark contemplative jazz for late-night":
  "quiet piano, muted trumpet, soft brushed drums, dark atmosphere,
  slow tempo, smoky room, intimate recording"
- refinement: what CHANGED in the LAST message from previous turns.
  "wants more piano focus", "less electronic, more acoustic",
  "darker and more mysterious than last track"
  This is critical for understanding the direction shift.
- rejected: tracks or styles user said don't fit the mood.
  Include the reason: "too upbeat", "too electronic for contemplative mood",
  "not acoustic enough"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "detailed atmosphere and setting keywords",
  "clap_keywords": "sonic translation of the atmosphere",
  "refinement": "what changed in this turn" or null,
  "rejected": []
}}"""


# D-LH: 기억 속 OST/영화 곡 찾기
PROMPT_LH = """You are helping find a specific soundtrack track the user remembers.

The user has a song from a movie, game, or TV show in mind but can't recall
the exact title. They describe HOW IT FEELS or what SCENE it fits.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: the COMPOSER if mentioned or confirmed.
  Example: "Brian Tyler", "Howard Shore", "Mark Mancina", "Ramin Djawadi"
  If unknown, set to null.
- source: the MOVIE, GAME, or SHOW the soundtrack is from.
  Example: "Moana", "Assassin's Creed IV Black Flag", "Game of Thrones"
  Critical for narrowing the search.
- tag_list: combine source + genre + mood for metadata matching.
  Example: "Moana, orchestral, epic, adventure, sailing, grand discovery,
  Mark Mancina, animated film soundtrack"
  Include the source name in tag_list — it's likely in the metadata.
- clap_keywords: the FEELING the user describes about the music.
  Focus on emotional/atmospheric words:
  "sweeping orchestral, building to a climax, grand and adventurous,
  sense of destiny, ocean calling, spiritual feeling"
- found: true if user confirmed finding the track.
  After found, user typically asks for "more from this soundtrack"
  or "similar from a different movie".
- wants_different_source: true if user says "from a different movie/game"
  or "other composers". false if still exploring same source.
- rejected: tracks that were close but not right.
  Include why: "too intense, looking for more journey-focused",
  "right era but wrong feel"

Output ONLY valid JSON:
{{
  "artist_name": "composer name" or null,
  "source": "movie/game/show name" or null,
  "tag_list": "source + genre + mood for metadata",
  "clap_keywords": "emotional/atmospheric feeling descriptors",
  "found": false,
  "wants_different_source": false,
  "rejected": []
}}"""


# D-LL: 아티스트 자유 탐색
PROMPT_LL = """You are extracting artist exploration keywords for casual music discovery.

The user is casually exploring a specific artist's music across different
moods and styles, or discovering music for general situations.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: the PRIMARY artist being explored.
  If user is still focused on one artist: keep that artist.
  Example: exploring Red Hot Chili Peppers → "Red Hot Chili Peppers"
  If user says "other artists too" or "similar to [artist]",
  keep the reference artist but note wants_different_artist.
- tag_list: what MOOD or STYLE the user wants from the LAST message.
  This changes each turn as they explore different facets:
  "chill, relaxing" → "upbeat, funky, dancing" → "strong bassline, Flea"
  → "epic, anthemic, big" → "intense, heavy rock"
  Extract ONLY from the LAST message.
- clap_keywords: sonic translation of what user wants.
  "chill out" → "mellow guitar, laid-back rhythm, smooth vocals"
  "funky dancing" → "groovy bass, upbeat drums, energetic"
  "epic anthemic" → "big guitar riffs, stadium rock, powerful chorus"
- wants_different_artist: true if user asks for other artists.
- vibe_label: one-word summary of current mood request.
  "chill" | "energetic" | "heavy" | "emotional" | "funky" | "epic"

Output ONLY valid JSON:
{{
  "artist_name": "primary artist",
  "tag_list": "current mood/style from LAST message",
  "clap_keywords": "sonic translation of mood",
  "wants_different_artist": false,
  "vibe_label": "one-word mood summary",
  "rejected": []
}}"""


PROMPTS = {
    "HH": PROMPT_HH,
    "HL": PROMPT_HL,
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "default": PROMPT_HL,  # D에서 specificity 미상이면 HL이 가장 무난 (28세션으로 최다)
}


# ── 쿼리 빌더 ─────────────────────────────────────────────────────────────────

def _build_bge_query(data: dict, specificity: str) -> str:
    """BGE 메타데이터 인덱스 검색용 쿼리 문자열 구성."""
    parts = []
    if data.get("artist_name"):
        parts.append(f"artist_name: {data['artist_name']}")
    if specificity == "LH" and data.get("source"):
        # OST는 소스명(영화/게임명)이 메타데이터에 포함될 가능성 높음
        parts.append(f"album_name: {data['source']}")
    if data.get("tag_list"):
        parts.append(f"tag_list: {data['tag_list']}")
    return "\n".join(parts)


def build_result(data: dict, specificity: str, fallback: str) -> dict:
    """JSON 파싱 결과를 retriever가 쓰는 형식으로 변환."""
    bge_query     = _build_bge_query(data, specificity) or fallback
    clap_keywords = data.get("clap_keywords") or data.get("tag_list") or fallback
    attr_query    = data.get("tag_list") or fallback

    result = {
        "direct_request": data.get("direct_request"),
        "bge_query":      bge_query,
        "clap_keywords":  clap_keywords,
        "attr_query":     attr_query,
        "rejected":       data.get("rejected") or [],
    }
    # 카테고리 전용 필드 그대로 전달
    for key in ("found", "source", "wants_different_source",
                "wants_different_artist", "frustrated", "refinement", "vibe_label"):
        if key in data:
            result[key] = data[key]
    return result
