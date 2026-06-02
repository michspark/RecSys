"""Category J extraction prompts — 인기곡/히트곡 탐색.

77세션 (8%) — popularity score가 결정적 신호인 카테고리.
유저가 "popular", "famous", "well-known", "iconic" 등을 명시적으로 요청.

Specificity 분포: HH=10, HL=30, LH=16, LL=21
"""

# J-HH: 정확한 인기곡 요청
PROMPT_HH = """You are extracting exact popular track requests from a music conversation.

The user is requesting a SPECIFIC well-known/popular song by title and/or artist,
or asking for the MOST popular song by a specific artist.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- direct_request: if user names an exact song to play.
  "Play 'Heart-Shaped Box' by Nirvana"
  → {{"track_name": "Heart-Shaped Box", "artist_name": "Nirvana"}}
  "Which ONE OK ROCK song is most popular?" → null (no exact track named)
- artist_name: the artist in focus.
- tag_list: genre + era + popularity context.
  Include popularity indicators the user mentioned:
  "highly popular", "grunge", "90s", "iconic"
  Example: "grunge, 90s, highly popular, Nirvana, iconic"
- popularity_context: WHY the user considers this popular.
  "chart hit", "everyone knows it", "defining track of the era",
  "most popular by this artist", "fan favorite"
- scope: what user wants after finding the track.
  "more_popular_same_genre": other popular tracks in same genre
  "more_by_artist": more from the same artist
  "popular_different_style": popular tracks but different genre
  "most_popular_by_artist": ranking tracks by one artist's popularity

Output ONLY valid JSON:
{{
  "direct_request": {{...}} or null,
  "artist_name": "...",
  "tag_list": "genre + era + popularity descriptors",
  "popularity_context": "why user considers this popular",
  "scope": "more_by_artist",
  "rejected": []
}}"""


# J-HL: 니치 커뮤니티 내 인기곡 탐색
PROMPT_HL = """You are extracting niche-community popularity keywords from a music conversation.

The user wants tracks that are popular within SPECIFIC music communities
or subgenres — not mainstream pop hits, but tracks beloved by dedicated fans
of a particular scene.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: if user mentions or reacts to a specific artist.
  If browsing across artists in a scene, set to null.
- tag_list: the SCENE or COMMUNITY + current style direction.
  ALWAYS include the niche/community identifier:
  "outlaw country fans", "indie electro scene", "dark synthpop community",
  "classic R&B fan favorites", "neo-soul cult classics"
  PLUS current refinement from LAST message:
  "outlaw country, gritty storytelling" → "indie electro, trending"
  → "dark synthpop, atmospheric" → "electro-industrial, driving bassline"
- clap_keywords: how this niche's sound is distinctive.
  "outlaw country" → "raw twangy guitar, deep male vocals,
  rebellious lyrics, lo-fi country production"
  "dark synthpop" → "haunting female vocals, rich synth textures,
  atmospheric pads, moody bassline"
- niche_community: the specific music community.
  "outlaw country", "indie electro", "dark synthpop",
  "underground R&B", "neo-soul", "electro-industrial"
- popularity_type: what kind of popularity matters here.
  "scene_classic": beloved within the niche community
  "cult_following": not mainstream but dedicated fan base
  "trending": currently popular in the scene
  "deep_cut": lesser known but respected by connoisseurs
- rejected: what didn't fit the niche.
  "too mainstream", "not dark enough for synthpop fans",
  "wrong subgenre within the scene"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "niche community + current style",
  "clap_keywords": "distinctive sound of the niche",
  "niche_community": "specific community or scene",
  "popularity_type": "scene_classic",
  "rejected": []
}}"""


# J-LH: 기억 속 히트곡 찾기
PROMPT_LH = """You are helping find a specific hit song from vague memory.

The user remembers a popular song but can't recall exact details.
They know it was a HIT — widely played, everyone knew it — and give
clues about era, genre, and feel.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: confirmed or guessed artist.
  "YES! Smells Like Teen Spirit!" → "Nirvana"
  Still searching: "maybe from the 90s, East Coast hip-hop"
- tag_list: cumulative clues about the hit song.
  Build from ALL turns, dropping rejected clues:
  "early 90s, everyone knew it, really big"
  → "early 90s, big hit, hip-hop, hard beat"
  → "90s, East Coast, hip-hop, gritty, boom-bap"
  ALWAYS include popularity indicators: "big hit", "widely known",
  "everyone knew it", "played everywhere"
- popularity_era: the specific era when this was popular.
  "early 90s", "2013-2015", "late 2000s", "80s"
- found: true if user confirmed finding the song.
  After found, user typically shifts to "more hits from same era/genre".
- mood_clue: emotional/energy feel of the remembered hit.
  "upbeat, feel-good", "hard-hitting", "anthemic",
  "danceable", "everyone sang along"
- rejected: songs that were hits but not THE one.
  Include why: "right era but wrong genre",
  "good song but not the one I'm thinking of"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "cumulative clues + popularity indicators",
  "popularity_era": "when this was popular",
  "found": false,
  "mood_clue": "energy/feel of the hit",
  "rejected": []
}}"""


# J-LL: 넓은 인기곡 브라우징
PROMPT_LL = """You are extracting broad popular music browsing keywords.

The user wants well-known, widely popular songs from a genre or era.
No specific track in mind — just "give me the hits."

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: if user reacted positively to an artist.
  For general hit browsing, usually null.
- tag_list: era + genre + popularity level from LAST message.
  Keep it simple — match the user's broad request:
  "90s, popular, party songs, nostalgic"
  "80s, dance-pop, well-known, chart hits"
  "current, trending, R&B, female vocalist"
  ALWAYS include a popularity indicator.
- popularity_era: the target era.
  "90s", "80s", "2010s", "current/trending"
- energy_preference: what kind of hits they want.
  "party": upbeat, dance, fun
  "nostalgic": classic hits that bring back memories
  "chill": smooth, laid-back popular tracks
  "any": no specific energy preference
- genre_focus: the broad genre if specified.
  "hip-hop", "dance-pop", "R&B", "rock", "general pop"
  Set to null if user says "any genre" or doesn't specify.
- rejected: what didn't fit.
  "too obscure", "not upbeat enough", "wrong era"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "era + genre + popularity indicators",
  "popularity_era": "target era",
  "energy_preference": "any",
  "genre_focus": "..." or null,
  "rejected": []
}}"""


PROMPTS = {
    "HH": PROMPT_HH,
    "HL": PROMPT_HL,
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "default": PROMPT_HL,  # J에서 specificity 미상이면 HL (30세션으로 최다)
}


# ── 쿼리 빌더 ─────────────────────────────────────────────────────────────────

def _build_bge_query(data: dict, specificity: str) -> str:
    """BGE 메타데이터 인덱스 검색용 쿼리 문자열 구성."""
    parts = []
    if data.get("artist_name"):
        parts.append(f"artist_name: {data['artist_name']}")
    # J-HL: niche_community를 별도 tag_list 행으로 추가
    if specificity == "HL" and data.get("niche_community"):
        parts.append(f"tag_list: {data['niche_community']}")
    if data.get("tag_list"):
        parts.append(f"tag_list: {data['tag_list']}")
    return "\n".join(parts)


def build_result(data: dict, specificity: str, fallback: str) -> dict:
    """JSON 파싱 결과를 retriever가 쓰는 형식으로 변환.

    use_popularity=True: retriever가 popularity_scores를 블렌드에 포함시킴.
    """
    bge_query     = _build_bge_query(data, specificity) or fallback
    clap_keywords = data.get("clap_keywords") or data.get("tag_list") or fallback
    attr_query    = data.get("tag_list") or fallback

    result = {
        "direct_request":   data.get("direct_request"),
        "bge_query":        bge_query,
        "clap_keywords":    clap_keywords,
        "attr_query":       attr_query,
        "rejected":         data.get("rejected") or [],
        "use_popularity":   True,   # J는 모든 specificity에서 popularity 활성화
    }
    # 카테고리 전용 필드 그대로 전달
    for key in ("found", "popularity_era", "popularity_context",
                "niche_community", "popularity_type",
                "energy_preference", "genre_focus", "scope", "mood_clue"):
        if key in data:
            result[key] = data[key]
    return result
