"""Category E extraction prompts — 기술적 매칭 / 음악 여정 / 취향 발견 / 자유 탐색."""

# E-HH: 정확한 기술적 특성 매칭
PROMPT_HH = """You are extracting precise technical music characteristics from a conversation.

The user is searching for tracks with EXACT technical properties:
key signature, tempo/BPM, album, production style, or sonic architecture.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: reference artist or target artist if specified.
  Example: "Aphex Twin" as the artist being explored.
- album_name: specific album if mentioned.
  Example: "Drukqs", "Selected Ambient Works 85-92"
- tag_list: technical and stylistic descriptors for metadata matching.
  Include: subgenre (glitchy, breakcore, ambient, industrial),
  era, and any production characteristics.
  Example: "experimental electronic, glitchy, breakcore, aggressive,
  Aphex Twin, Drukqs, dark industrial"
- technical_specs: any exact musical specifications mentioned.
  Key: "D# minor", "F minor", "C# minor"
  Tempo: "118 BPM", "79.23 BPM", "around 89 BPM"
  Other: "same key as Avril 14th"
  Set to null if no exact specs in LAST message.
- search_mode: what the user is doing NOW in the LAST message.
  "exact_match": still looking for precise technical matches
  "similar_mood": gave up on exact specs, wants similar mood instead
  "exploring_artist": browsing the artist's catalog freely
  This changes as user relaxes constraints across turns.
- rejected: tracks that didn't match specs.
  Include why: "wrong key", "close but D minor not C# minor"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "album_name": "..." or null,
  "tag_list": "technical and stylistic descriptors",
  "technical_specs": {{"key": "...", "tempo": "..."}} or null,
  "search_mode": "exact_match",
  "rejected": []
}}"""


# E-HL: 음악적 여정 (장르 프로그레션)
PROMPT_HL = """You are tracking a PLANNED musical journey across turns.

The user wants to progressively move through musical styles,
moods, or intensities — like a guided tour through genres.
Each turn is a planned step in the journey.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: if user asks for a specific artist or composer.
  If user says "different artist" or "other composers", set to null.
- tag_list: where the user is NOW in their journey.
  NOT the starting point, but the CURRENT destination from the LAST message.
  Example journey: EBM → futurepop → melodic synthpop → melancholic → uptempo
  If last message says "now something more melancholic":
  tag_list = "melodic synthpop, melancholic, emotional depth, futurepop"
- clap_keywords: how the CURRENT stage should SOUND.
  Example for "melancholic but melodic synthpop":
  "polished synths, emotional vocals, driving beat, melancholic melody,
  clean production, atmospheric"
- journey_stage: describe where user is in their journey.
  "early": still at the starting genre/style
  "middle": transitioning between styles
  "late": reaching the destination or refining the final style
  "complete": user said they're done or wants something different
- direction: what CHANGE the user requested in the LAST message.
  "more melodic", "darker", "more uptempo", "more atmospheric",
  "different artist", "from a different game"
  This is the DELTA from the previous recommendation.
- constraints: things user wants to KEEP across the journey.
  "always melodic, never harsh", "keep female vocals",
  "stay instrumental", "no harsh industrial"
- rejected: what didn't fit the current stage.
  "too aggressive for this stage", "that's going backwards"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "current destination style descriptors",
  "clap_keywords": "how current stage should sound",
  "journey_stage": "early",
  "direction": "requested change from last turn" or null,
  "constraints": "what to always keep or avoid" or null,
  "rejected": []
}}"""


# E-LH: 취향 자기 발견 (가장 많은 패턴, 44세션)
PROMPT_LH = """You are helping a user discover WHAT they like about music.

The user started with a song they love but couldn't explain WHY.
Through conversation, they are gradually discovering specific musical
elements that appeal to them. There is a key REALIZATION moment.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: the seed artist, or current artist being discussed.
  Example: started with "Take Five" → artist_name = "Dave Brubeck"
  After realization: might shift to various artists.
- seed_track: the ORIGINAL song the user is trying to understand.
  This stays the same throughout the conversation.
  Example: "Take Five", "Duality by Slipknot", "Aunt Leslie by Vulfpeck"
- tag_list: depends on the discovery phase:
  BEFORE realization: broad descriptors around the seed track.
  "cool jazz, saxophone, relaxed, Dave Brubeck, Take Five"
  AFTER realization: the DISCOVERED specific elements.
  "unusual time signature, smooth saxophone melody, rhythmic playfulness"
- discovered_elements: what the user has figured out so far.
  Build this CUMULATIVELY across turns:
  Turn 2: "unusual rhythm"
  Turn 4: "unusual rhythm + smooth saxophone"
  Turn 6: "unusual time signature + hypnotic sax melody = what I love"
  Extract from user's "YES!", "that's it!", "I've realized" statements.
- discovery_phase: where the user is in their journey.
  "exploring": trying different aspects, hasn't found it yet
  "narrowing": getting closer, some elements identified
  "eureka": the realization moment ("YES! That's exactly it!")
  "applying": found what they like, now looking for more with same elements
- clap_keywords: depends on phase.
  exploring: broad sonic descriptors of the seed track
  eureka/applying: the SPECIFIC sonic elements discovered
  Example after eureka: "5/4 time signature feel, looping saxophone melody,
  laid-back groove, rhythmic complexity with smooth melody"
- rejected: what was tried but didn't capture the appeal.
  "energetic brass wasn't it", "too melodic, missing the rhythm"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "seed_track": "original song being analyzed" or null,
  "tag_list": "phase-appropriate descriptors",
  "discovered_elements": "cumulative discovered preferences" or null,
  "discovery_phase": "exploring",
  "clap_keywords": "phase-appropriate sonic descriptors",
  "rejected": []
}}"""


# E-LL: 자유 탐색, 새 음악 발견
PROMPT_LL = """You are extracting discovery-oriented keywords from a casual music exploration.

The user has no specific song or style in mind. They want to DISCOVER
new music and are open to different genres and artists.
Their preferences emerge from reactions to recommendations.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: an artist the user reacted positively to.
  If they say "I like this artist!", include them.
  If they say "something completely different", set to null.
  If they say "more like [artist]", include that artist.
- tag_list: what the user wants NOW based on their LAST message.
  This evolves as they discover what they like:
  "fresh, underground hip-hop" → "experimental, electronic textures"
  → "atmospheric electronic" → "chill, ambient, downtempo"
  Extract ONLY from the LAST message.
- clap_keywords: sonic description of what they want now.
  "atmospheric electronic, subtle beats, immersive soundscape,
  ambient textures, serene, spacious"
- emerging_preference: what pattern is forming across turns.
  Look at what user consistently likes and dislikes:
  "gravitating toward atmospheric/ambient electronic"
  "likes experimental texture, dislikes mainstream"
  "prefers chill over energetic"
  This helps predict what to recommend next.
- wants_new_artist: true if user explicitly wants artists they haven't
  heard in this session. false if happy hearing more from current artist.
- rejected: what didn't work and why.
  "too mainstream", "already knew that one", "too energetic"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "current discovery direction from LAST message",
  "clap_keywords": "sonic description of current interest",
  "emerging_preference": "pattern forming across turns" or null,
  "wants_new_artist": false,
  "rejected": []
}}"""


PROMPTS = {
    "HH": PROMPT_HH,
    "HL": PROMPT_HL,
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "default": PROMPT_LH,  # E에서 specificity 미상이면 LH (44세션으로 최다)
}


# ── 쿼리 빌더 ─────────────────────────────────────────────────────────────────

def _compute_dynamic_weights(data: dict, specificity: str) -> dict | None:
    """Specificity + phase/stage에 따라 동적 가중치 계산.

    반환값: {"metadata": float, "attributes": float, "audio": float}
    retriever가 이 값을 읽어서 score blend를 조정.
    None이면 retriever가 기본 가중치를 사용.
    """
    if specificity == "LH":
        # discovery_phase에 따라 BGE(metadata) vs attr 비중이 역전됨
        phase = data.get("discovery_phase", "exploring")
        weights = {
            "exploring": {"metadata": 0.7, "attributes": 0.1, "audio": 0.2},
            "narrowing":  {"metadata": 0.5, "attributes": 0.3, "audio": 0.2},
            "eureka":     {"metadata": 0.3, "attributes": 0.5, "audio": 0.2},
            "applying":   {"metadata": 0.3, "attributes": 0.4, "audio": 0.3},
        }
        return weights.get(phase)

    if specificity == "HL":
        # journey_stage에 따라 audio(CLAP) 비중이 점점 올라감
        stage = data.get("journey_stage", "early")
        weights = {
            "early":    {"metadata": 0.6, "attributes": 0.2, "audio": 0.2},
            "middle":   {"metadata": 0.4, "attributes": 0.3, "audio": 0.3},
            "late":     {"metadata": 0.3, "attributes": 0.3, "audio": 0.4},
            "complete": {"metadata": 0.3, "attributes": 0.3, "audio": 0.4},
        }
        return weights.get(stage)

    return None


def _build_bge_query(data: dict) -> str:
    """BGE 메타데이터 인덱스 검색용 쿼리 문자열 구성."""
    parts = []
    if data.get("artist_name"):
        parts.append(f"artist_name: {data['artist_name']}")
    if data.get("album_name"):
        parts.append(f"album_name: {data['album_name']}")
    if data.get("tag_list"):
        parts.append(f"tag_list: {data['tag_list']}")
    return "\n".join(parts)


def build_result(data: dict, specificity: str, fallback: str) -> dict:
    """JSON 파싱 결과를 retriever가 쓰는 형식으로 변환."""
    bge_query     = _build_bge_query(data) or fallback
    clap_keywords = data.get("clap_keywords") or data.get("tag_list") or fallback

    # E-LH: eureka/applying 단계에서 discovered_elements를 attr 쿼리에 사용
    if specificity == "LH" and data.get("discovered_elements"):
        attr_query = data["discovered_elements"]
    else:
        attr_query = data.get("tag_list") or fallback

    dynamic_weights = _compute_dynamic_weights(data, specificity)

    result = {
        "direct_request":   None,
        "bge_query":        bge_query,
        "clap_keywords":    clap_keywords,
        "attr_query":       attr_query,
        "dynamic_weights":  dynamic_weights,  # retriever가 읽어서 가중치 동적 적용
        "rejected":         data.get("rejected") or [],
    }
    # 카테고리 전용 필드 그대로 전달
    for key in ("seed_track", "discovered_elements", "discovery_phase",
                "journey_stage", "direction", "constraints",
                "technical_specs", "search_mode", "emerging_preference",
                "wants_new_artist", "album_name"):
        if key in data:
            result[key] = data[key]
    return result
