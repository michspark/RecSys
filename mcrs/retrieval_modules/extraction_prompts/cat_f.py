"""Category F extraction prompts — 정확한 버전 찾기 / 장르+시대 탐색 / 기억 속 곡 / 장르 자유 탐색."""

# F-HH: 정확한 곡/앨범/버전 찾기
PROMPT_HH = """You are extracting exact track identification from a music conversation.

The user is looking for a SPECIFIC song, often from a specific album or version.
After finding it, they explore more from the same artist or album.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- direct_request: if the LAST message asks to play a specific song.
  Include track_name, artist_name, and album_name if specified.
  Example: "Play 'Wait For It' from Hamilton Original Broadway Cast Recording"
  → {{"track_name": "Wait For It", "artist_name": "Lin-Manuel Miranda",
      "album_name": "Hamilton Original Broadway Cast Recording"}}
  Set to null if user is browsing, not requesting a specific track.
- artist_name: the artist/composer currently being explored.
- album_name: if user is exploring tracks from a SPECIFIC album.
  Example: "What other songs are on Hybrid Theory?" → "Hybrid Theory"
  This is critical for F-HH — many sessions stay within one album.
- tag_list: genre and style for metadata matching.
  Example: "nu-metal, alternative rock, Linkin Park, Hybrid Theory, 2000"
  Include album name and era in tag_list.
- version_info: if user asks for a specific VERSION or recording.
  "Original Broadway Cast Recording", "2003 release", "demo version",
  "Mixtape version", "remastered"
  Set to null if no version specified.
- scope: where user wants to search now.
  "same_album": more tracks from the same album
  "same_artist": other albums by same artist
  "similar_artist": different artist, similar style
  "specific_track": requesting one exact track

Output ONLY valid JSON:
{{
  "direct_request": {{"track_name": "...", "artist_name": "...", "album_name": "..."}} or null,
  "artist_name": "...",
  "album_name": "..." or null,
  "tag_list": "genre, style, album, era descriptors",
  "version_info": "..." or null,
  "scope": "specific_track",
  "rejected": []
}}"""


# F-HL: 장르+시대 내 여러 곡
PROMPT_HL = """You are extracting genre-era specific search keywords from a music conversation.

The user is deep-diving into a SPECIFIC genre + time period combination,
exploring multiple tracks within those boundaries. They refine within
the genre each turn but stay in the same era.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: if user asks for a specific artist or says "like [artist]".
  For genre browsing without a specific artist, set to null.
  If user says "by a different artist", set to null.
- tag_list: the GENRE + ERA + CURRENT REFINEMENT from the LAST message.
  ALWAYS include:
  1. The core genre: "hard bop", "electronic dance", "hardcore punk"
  2. The era: "late 50s", "early 2000s", "80s"
  3. The CURRENT refinement: what's different about THIS turn's request
  Example progression:
  T1: "late 90s electronic, energetic, dance party"
  T3: "late 90s electronic, experimental, ambient"
  T5: "late 90s electronic, deep bassline, atmospheric"
  T7: "late 90s electronic, driving house, strong beat"
  The genre+era stays constant, refinement changes.
- clap_keywords: how the CURRENT refinement should SOUND.
  Example for "late 90s electronic with deep bassline":
  "deep thumping bass, atmospheric pads, 90s production,
  warm analog synths, rhythmic, dark undertones"
- sub_refinement: what SPECIFIC aspect the user is exploring NOW.
  "melodic focus", "experimental edge", "political lyrics",
  "saxophone solos", "strong bassline", "vocal jazz",
  "more abrasive and noisy", "uptempo driving beat"
  Extract from the LAST message only.
- era_locked: the time period that stays constant.
  "late 90s to early 2000s", "50s-60s", "80s", "mid-2000s"
- genre_locked: the core genre that stays constant.
  "electronic", "hard bop jazz", "hardcore punk", "progressive rock"
- rejected: what didn't fit the genre/era combo.
  Include why: "too recent", "that's bebop not swing",
  "too melodic, need more abrasive"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "genre + era + current refinement",
  "clap_keywords": "sonic description of current refinement",
  "sub_refinement": "specific aspect being explored now" or null,
  "era_locked": "constant time period" or null,
  "genre_locked": "constant core genre" or null,
  "rejected": []
}}"""


# F-LH: 기억 속 곡 찾기 (장르/시대 단서 + 가사)
PROMPT_LH = """You are helping find a specific track the user remembers,
using genre, era, and sometimes lyrical/thematic clues.

The user has a song in mind and is narrowing down through genre, era,
instrument, and sometimes lyrical theme clues.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: any artist confirmed or guessed.
  If user says "YES! Roy Ayers!", include the confirmed artist.
  If still searching: "maybe Donald Byrd or Roy Ayers Ubiquity"
- tag_list: ALL clues combined for metadata matching.
  Genre: "jazz-funk", "reggae", "progressive rock"
  Era: "70s", "late 80s"
  Instrument: "vibraphone", "saxophone", "acoustic guitar"
  Theme: "metropolitan", "Rasta culture", "herb theme"
  Build CUMULATIVELY — include clues from previous turns that
  haven't been rejected.
  Example: "70s jazz-funk, vibraphone, metropolitan feel,
  Roy Ayers Ubiquity, instrumental, groovy"
- lyric_keywords: if the user mentions lyrical THEMES or CONTENT.
  "about Rasta culture", "songs about smoking ganja",
  "positive message", "social commentary"
  Set to null if user describes only the SOUND, not lyrics.
- clap_keywords: sonic description from the user's clues.
  "laid-back groove, vibraphone melody, funky bassline,
  70s production, warm, metropolitan jazz feel"
- found: true if user confirmed finding the track.
  After found, keywords should shift to "similar to [found track]".
- instrument_focus: specific instruments the user highlights.
  "vibraphone", "saxophone solo", "prominent brass",
  "strong bassline", "acoustic guitar"
  Set to null if no specific instrument mentioned.
- rejected: tracks that were close but wrong.
  Include what was wrong: "right era but not the one",
  "has vocals, need instrumental", "too modern"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "cumulative genre + era + instrument + theme clues",
  "lyric_keywords": "thematic/lyrical clues" or null,
  "clap_keywords": "sonic description from all clues",
  "found": false,
  "instrument_focus": "..." or null,
  "rejected": []
}}"""


# F-LL: 장르 자유 탐색
PROMPT_LL = """You are extracting broad genre exploration keywords from a music conversation.

The user is casually exploring a genre's range — different moods,
eras, artists, and subgenres within it. No specific target track.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: if user mentions or reacts to a specific artist.
  "Miles Davis is always a pleasure" → "Miles Davis"
  If user wants variety: "different artist" → set to null.
- tag_list: the CURRENT exploration direction from the LAST message.
  The user moves through different facets of the genre:
  "classic jazz, mellow, relaxed" → "swing, faster tempo"
  → "bluesy, soulful" → "trumpet solos" → "piano solos"
  Extract ONLY the LAST message's direction, but keep the base genre.
  Example: "classic jazz, piano solos, exceptional"
- clap_keywords: how the current direction should SOUND.
  "mellow trumpet, brushed drums, laid-back bass, smoky atmosphere"
  → "swinging rhythm, energetic brass, uptempo drums"
  → "deep piano chords, reflective, solo spotlight"
- exploration_facet: what ASPECT of the genre user wants to explore now.
  "different mood" | "different era" | "different instrument" |
  "different artist" | "different subgenre" | "general browsing"
- base_genre: the CORE genre that stays constant throughout.
  "classic jazz", "reggae", "classic rock"
  This doesn't change even as sub-explorations shift.
- rejected: what user found too different or not fitting.
  "too modern for classic jazz", "too electronic"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "base genre + current exploration direction",
  "clap_keywords": "sonic description of current direction",
  "exploration_facet": "what aspect is being explored" or null,
  "base_genre": "constant core genre" or null,
  "rejected": []
}}"""


PROMPTS = {
    "HH": PROMPT_HH,
    "HL": PROMPT_HL,
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "default": PROMPT_HL,  # F에서 specificity 미상이면 HL (32세션으로 최다)
}


# ── 쿼리 빌더 ─────────────────────────────────────────────────────────────────

def _build_bge_query(data: dict, specificity: str) -> str:
    """BGE 메타데이터 인덱스 검색용 쿼리 문자열 구성."""
    parts = []

    if specificity == "HH":
        # 정확한 매칭: 아티스트 + 앨범 + 태그
        if data.get("artist_name"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("album_name"):
            parts.append(f"album_name: {data['album_name']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")

    elif specificity == "HL":
        # 장르+시대 고정: tag_list에 genre+era+refinement가 이미 합쳐져 있음
        if data.get("artist_name"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")

    elif specificity == "LH":
        # 기억 속 곡: 누적 단서 전부 + 악기 포커스
        if data.get("artist_name"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")
        # instrument_focus가 있으면 별도 tag_list 행으로 추가
        if data.get("instrument_focus"):
            parts.append(f"tag_list: {data['instrument_focus']}")

    else:  # LL
        if data.get("artist_name"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")

    return "\n".join(parts)


def _build_attr_query(data: dict, specificity: str) -> str:
    """attributes-qwen3 임베딩 검색용 쿼리 구성."""
    if specificity == "HL" and data.get("sub_refinement"):
        # refinement이 있으면 그걸 앞에 배치해 attr 검색에서 현재 방향을 강조
        return f"{data['sub_refinement']}, {data.get('tag_list', '')}"
    return data.get("tag_list") or ""


def build_result(data: dict, specificity: str, fallback: str) -> dict:
    """JSON 파싱 결과를 retriever가 쓰는 형식으로 변환."""
    bge_query     = _build_bge_query(data, specificity) or fallback
    clap_keywords = data.get("clap_keywords") or data.get("tag_list") or fallback
    attr_query    = _build_attr_query(data, specificity) or fallback

    # F-LH: lyric_keywords가 있으면 lyrics-qwen3 검색에 활용
    lyrics_query = data.get("lyric_keywords") if specificity == "LH" else None

    result = {
        "direct_request": data.get("direct_request"),
        "bge_query":      bge_query,
        "clap_keywords":  clap_keywords,
        "attr_query":     attr_query,
        "lyrics_query":   lyrics_query,
        "rejected":       data.get("rejected") or [],
    }
    # 카테고리 전용 필드 그대로 전달
    for key in ("found", "album_name", "scope", "version_info",
                "era_locked", "genre_locked", "sub_refinement",
                "instrument_focus", "base_genre", "exploration_facet"):
        if key in data:
            result[key] = data[key]
    return result
