"""Category H extraction prompts — 정확한 곡 재생 / 아티스트+서브장르 탐색 / 스타일로 기억 찾기 / 열린 탐색."""

# H-HH (14세션): 정확한 곡 재생 요청
PROMPT_HH = """You are extracting exact track requests from a music conversation.

The user requests SPECIFIC songs by exact title and artist.
After finding them, they typically keep asking for more from the SAME artist.
This pattern often continues for the entire session.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- direct_request: if LAST message asks to play a specific song.
  {{"track_name": "...", "artist_name": "..."}}
  Triggers: "play", "put on", "can you find", "play another [artist] song"
  IMPORTANT: "Play another Weird Al song" is NOT a direct_request
  (no specific track named). Only set when a TITLE is given.
- artist_name: the artist the user is focused on.
  H-HH sessions often stay on ONE artist the entire time:
  8 turns of Weird Al, 8 turns of Pantera, 8 turns of Fila Brazillia.
  Keep this consistent unless user explicitly switches.
- album_name: if user asks for tracks from a specific album.
  "from the album 'Dicks'" → "Dicks"
  "from their earlier albums" → null (too vague for exact album)
- tag_list: genre/style for when direct match fails.
  "comedy, parody, novelty" for Weird Al
  "groove metal, heavy, crushing riffs" for Pantera
  "downtempo, trip-hop, chill electronic" for Fila Brazillia
- same_artist_mode: true if user is requesting more from the SAME artist
  without naming a specific track. This is the dominant pattern in H-HH.
  "Play another Weird Al song" → true
  "Play 'Domination' by Pantera" → false (specific track)
- frustrated: true if user keeps getting the wrong track.
  Pattern: "No, you STILL haven't played Domination"
  When frustrated, direct_request is the ONLY thing that matters.

Output ONLY valid JSON:
{{
  "direct_request": {{"track_name": "...", "artist_name": "..."}} or null,
  "artist_name": "primary artist for entire session",
  "album_name": "..." or null,
  "tag_list": "genre, style fallback descriptors",
  "same_artist_mode": false,
  "frustrated": false,
  "rejected": []
}}"""


# H-HL (48세션): 아티스트 디스코그래피/서브장르 탐색
PROMPT_HL = """You are extracting artist/subgenre exploration keywords from a music conversation.

The user is systematically exploring a specific artist's catalog OR a narrow
subgenre. They stay focused on one anchor (artist or subgenre) and request
different facets, eras, or styles within it each turn.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: the PRIMARY anchor artist or band.
  If exploring one artist: "Green Day", "VNV Nation", "Sean Paul"
  If exploring a subgenre with no single artist: null
  If user says "different band but similar style", keep the REFERENCE
  artist but set wants_different_artist = true.
- tag_list: the CURRENT facet being explored from the LAST message.
  For artist exploration, this changes per turn:
  "early punk sound, 1039 Smoothed Out Slappy Hours" → "Dookie era, pop-punk"
  → "mature, American Idiot, political" → "emotional, angsty"
  For subgenre exploration:
  "80s hardcore punk, raw energy, short songs" → "political edge"
  → "melodic + metallic influences" → "chaotic, noisy, abrasive"
  Include era, album, and style qualifiers from the LAST message.
- clap_keywords: how the CURRENT facet should SOUND.
  "early Green Day" → "fast punk, raw recording, energetic, youthful"
  "mature Green Day" → "arena rock, polished production, anthemic"
  "chaotic hardcore" → "noisy, distorted, feedback, dissonant, aggressive"
- exploration_axis: WHAT aspect the user is exploring now.
  "different_era": same artist, different time period
  "different_album": same artist, specific album
  "different_facet": same artist, different style/mood
  "different_artist": wants a new artist in the same subgenre
  "deeper_niche": going more extreme/specific within the subgenre
- anchor_subgenre: the constant subgenre anchor if not artist-based.
  "80s American hardcore punk", "EBM/futurepop", "early 2000s dancehall"
  Set to null if the anchor is an artist, not a subgenre.
- wants_different_artist: true if user asks for other artists while
  staying in the same subgenre.
- rejected: what didn't fit the exploration scope.
  "that's bebop, I said swing", "too modern for 80s hardcore",
  "Fugazi is great but not chaotic enough"

Output ONLY valid JSON:
{{
  "artist_name": "anchor artist" or null,
  "tag_list": "current exploration facet from LAST message",
  "clap_keywords": "sonic description of current facet",
  "exploration_axis": "different_era",
  "anchor_subgenre": "constant subgenre" or null,
  "wants_different_artist": false,
  "rejected": []
}}"""


# H-LH (39세션): 기억 속 아티스트/곡 찾기
PROMPT_LH = """You are helping identify a specific artist or song from vague style descriptions.

The user remembers an artist or song by HOW IT SOUNDS or FEELS,
not the exact name. They give progressively more specific clues.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: any artist confirmed or being narrowed down.
  If confirmed: "YES! VNV Nation!" → "VNV Nation"
  If narrowing: based on clues, "maybe Method Man or Wu-Tang"
  After found, user often deep-dives into that artist's catalog.
- tag_list: cumulative style clues from ALL turns, dropping rejected ones.
  Build progressively:
  Turn 1: "electronic, powerful, anthemic"
  Turn 2: "electronic, powerful, anthemic, epic, introspective"
  Turn 3: "EBM, futurepop, epic, introspective, driving beat, VNV Nation"
  Include: genre, era, mood, sonic characteristics.
- clap_keywords: how the music SOUNDS based on user's descriptions.
  "powerful anthemic electronic" → "driving synth bass, soaring pads,
  epic build-ups, emotional male vocals, stadium-scale electronic"
  "raw 90s hip-hop with distinctive flow" → "gritty boom-bap beat,
  raw vocals, East Coast production, hard-hitting drums"
- found: true if user confirmed finding the artist/song.
  After found: extraction shifts to exploring that artist's catalog.
- post_found_mode: what user wants AFTER finding the target.
  "same_artist_deep_dive": more tracks from found artist
  "similar_artists": other artists with similar sound
  "specific_album": tracks from a particular album
  Set to null if not yet found.
- identity_clues: specific distinctive features user mentions.
  "almost anthemic sound", "distinctive vocal flow",
  "really stands out from typical rock", "grimy East Coast feel"
  These are the UNIQUE identifiers, not generic genre tags.
- rejected: artists/tracks that were close but wrong.
  Include why: "good but not anthemic enough",
  "right genre but too experimental", "wrong era"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "cumulative style clues",
  "clap_keywords": "sonic description from clues",
  "found": false,
  "post_found_mode": "same_artist_deep_dive" | "similar_artists" | "specific_album" or null,
  "identity_clues": "distinctive features mentioned" or null,
  "rejected": []
}}"""


# H-LL (34세션): 넓은 범위 음악 탐색
PROMPT_LL = """You are extracting open-ended music discovery keywords from a conversation.

The user started with broad exploration and their preferences are EMERGING
from reactions to recommendations. They might lock onto an artist or genre
as the conversation progresses.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: any artist the user has latched onto.
  If user says "I like EXID, more of them!" → "EXID"
  If still browsing broadly: null
  If user says "go back to [artist]" → that artist
- tag_list: what the user wants NOW from the LAST message.
  Track how preferences evolved:
  "new artists, fresh" → "hip-hop, underground" → "experimental electronic"
  → "atmospheric, chill" → "ambient, downtempo, serene"
  OR: "K-Pop, energetic" → "girl groups, 2010s" → "intense, powerful"
  Extract ONLY the LAST message's direction.
- clap_keywords: sonic description of current interest.
- locked_direction: if user has settled on a specific direction.
  "locked_artist": keeps asking for more from one artist
  "locked_genre": settled on a genre and wants variety within it
  "still_browsing": still open to different directions
  "returning": user said "go back to [previous thing]"
- genre_journey: brief summary of how genre shifted across turns.
  "started hip-hop → electronic → ambient" or "K-Pop throughout"
  Helps understand if user is genre-hopping or genre-locked.
- rejected: what user didn't like and why.
  "too different from K-Pop", "not what I'm looking for",
  "can we go back to [genre]?"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "current direction from LAST message",
  "clap_keywords": "sonic description of current interest",
  "locked_direction": "still_browsing",
  "genre_journey": "brief genre shift summary" or null,
  "rejected": []
}}"""


PROMPTS = {
    "HH": PROMPT_HH,
    "HL": PROMPT_HL,
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "default": PROMPT_HL,  # H에서 specificity 미상이면 HL (48세션으로 최다)
}


# ── 쿼리 빌더 ─────────────────────────────────────────────────────────────────

def _build_bge_query(data: dict, specificity: str) -> str:
    """BGE 메타데이터 인덱스 검색용 쿼리 문자열 구성."""
    parts = []

    if specificity == "HH":
        # 아티스트가 절대적 앵커 — album_name도 포함
        if data.get("artist_name"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("album_name"):
            parts.append(f"album_name: {data['album_name']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")

    elif specificity == "HL":
        # wants_different_artist이면 아티스트명 제외
        if data.get("artist_name") and not data.get("wants_different_artist"):
            parts.append(f"artist_name: {data['artist_name']}")
        # anchor_subgenre를 tag_list 앞에 별도 행으로 추가해 메타데이터 매칭 강화
        if data.get("anchor_subgenre"):
            parts.append(f"tag_list: {data['anchor_subgenre']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")

    elif specificity == "LH":
        # 누적 단서 + identity_clues를 별도 행으로
        if data.get("artist_name"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")
        if data.get("identity_clues"):
            parts.append(f"tag_list: {data['identity_clues']}")

    else:  # LL
        if data.get("artist_name"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")

    return "\n".join(parts)


def build_result(data: dict, specificity: str, fallback: str) -> dict:
    """JSON 파싱 결과를 retriever가 쓰는 형식으로 변환."""
    bge_query     = _build_bge_query(data, specificity) or fallback
    clap_keywords = data.get("clap_keywords") or data.get("tag_list") or fallback
    attr_query    = data.get("tag_list") or fallback

    # H-HH: same_artist_mode일 때 artist_name만으로 BGE 강화 신호
    same_artist_boost = (
        specificity == "HH"
        and bool(data.get("same_artist_mode"))
        and bool(data.get("artist_name"))
    )

    result = {
        "direct_request":    data.get("direct_request"),
        "bge_query":         bge_query,
        "clap_keywords":     clap_keywords,
        "attr_query":        attr_query,
        "rejected":          data.get("rejected") or [],
        "same_artist_boost": same_artist_boost,
    }
    # 카테고리 전용 필드 그대로 전달
    for key in ("found", "same_artist_mode", "frustrated",
                "exploration_axis", "anchor_subgenre", "wants_different_artist",
                "post_found_mode", "identity_clues",
                "locked_direction", "genre_journey"):
        if key in data:
            result[key] = data[key]
    return result
