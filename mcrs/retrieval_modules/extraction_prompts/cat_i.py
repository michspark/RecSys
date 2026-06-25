"""Category I extraction prompts — 국제/다문화 음악 탐색.

18세션 (2%) — 가장 작은 카테고리.
유저가 특정 문화권, 언어, 지역의 음악을 탐색하며
metadata-qwen3가 국가/언어/문화 태그를 잘 처리해 메타데이터 비중이 높음.

Specificity 분포: HH=3, HL=2, LH=8, LL=5
HH/HL은 세션 수가 너무 적어 하나의 프롬프트로 합침.
"""

# I-HH / I-HL (합침): 구체적인 아티스트/스타일 요청
PROMPT_SPECIFIC = """You are extracting international music search keywords.

The user is looking for music from a specific culture, language, or region,
often with exact artist/song requests or detailed style requirements.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- direct_request: if user names an exact song+artist to play.
  Example: "Play 'Africa' by Toto" → {{"track_name": "Africa", "artist_name": "Toto"}}
- artist_name: specific artist being explored or requested.
- tag_list: culture + genre + style descriptors.
  ALWAYS include: country/region, language, genre, era if mentioned.
  Example: "Brazilian, bossa nova, jazz fusion, 60s-70s, sophisticated"
  Example: "K-Pop, energetic, girl group, dance, high-energy"
  Example: "Eurovision, European pop, catchy, synth-pop, upbeat"
- culture_region: the specific culture or region.
  "Brazilian", "Korean/K-Pop", "Nordic/European", "Latin/bachata"
- wants_different_artist: true if user asks for new artists.

Output ONLY valid JSON:
{{
  "direct_request": {{"track_name": "...", "artist_name": "..."}} or null,
  "artist_name": "..." or null,
  "tag_list": "culture + genre + style",
  "culture_region": "culture or region",
  "wants_different_artist": false,
  "rejected": []
}}"""


# I-LH: 기억 속 국제 히트곡 찾기
PROMPT_LH = """You are helping find a specific international song from vague memory.

The user remembers a non-English or international hit but can't recall
the exact title. They give clues about language, era, and feel.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: confirmed or guessed artist.
  "YES! Shakira!" → "Shakira"
  Still searching: "maybe Shakira or similar Latin pop artist"
- tag_list: cumulative clues from all turns.
  Language: "non-English", "Spanish", "Korean"
  Era: "2015-2017", "early 2000s"
  Style: "danceable", "catchy beat", "female vocalist", "Latin pop"
  Build cumulatively, drop rejected clues.
- culture_region: "Latin", "European", "Korean", "Brazilian"
- found: true if user confirmed finding the song.
- rejected: songs/artists that weren't the one.

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "cumulative culture + style clues",
  "culture_region": "..." or null,
  "found": false,
  "rejected": []
}}"""


# I-LL: 넓은 문화 탐색
PROMPT_LL = """You are extracting cultural music exploration keywords.

The user is casually discovering music from different cultures.
No specific target — just exploring what sounds interesting.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: any artist user reacted positively to.
- tag_list: current cultural direction from LAST message.
  "K-Pop, high-energy, powerful" or "Brazilian funk, soul, 70s groove"
- culture_region: current region being explored.
- energy_preference: "high-energy" | "chill" | "any"
- wants_different_artist: true if user wants new artists.

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "culture + style from LAST message",
  "culture_region": "...",
  "energy_preference": "any",
  "wants_different_artist": false,
  "rejected": []
}}"""


PROMPTS = {
    "HH": PROMPT_SPECIFIC,
    "HL": PROMPT_SPECIFIC,   # 2세션이라 HH와 같은 프롬프트 사용
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "default": PROMPT_LH,    # I에서 specificity 미상이면 LH (8세션으로 최다)
}


# ── 쿼리 빌더 ─────────────────────────────────────────────────────────────────

def _build_bge_query(data: dict) -> str:
    """BGE 메타데이터 인덱스 검색용 쿼리 문자열 구성.

    culture_region을 별도 tag_list 행으로 추가 —
    메타데이터에 국가/지역 정보가 tag_list 컬럼에 있을 수 있음.
    """
    parts = []
    if data.get("artist_name"):
        parts.append(f"artist_name: {data['artist_name']}")
    if data.get("culture_region"):
        parts.append(f"tag_list: {data['culture_region']}")
    if data.get("tag_list"):
        parts.append(f"tag_list: {data['tag_list']}")
    return "\n".join(parts)


def build_result(data: dict, _specificity: str, fallback: str) -> dict:
    """JSON 파싱 결과를 retriever가 쓰는 형식으로 변환."""
    bge_query     = _build_bge_query(data) or fallback
    clap_keywords = data.get("tag_list") or fallback
    attr_query    = data.get("tag_list") or fallback

    result = {
        "direct_request": data.get("direct_request"),
        "bge_query":      bge_query,
        "clap_keywords":  clap_keywords,
        "attr_query":     attr_query,
        "rejected":       data.get("rejected") or [],
    }
    # 카테고리 전용 필드 그대로 전달
    for key in ("found", "culture_region", "wants_different_artist", "energy_preference"):
        if key in data:
            result[key] = data[key]
    return result
