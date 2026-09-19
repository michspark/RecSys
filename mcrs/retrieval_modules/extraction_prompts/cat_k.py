"""Category K extraction prompts — open exploration / discovery.

156 sessions (16%) — the largest category.
The user explores broadly without strict constraints; metadata + popularity + attributes are used evenly.

Specificity distribution: HH=15, HL=44, LH=47, LL=50
"""

# K-HH: finding representative tracks of a specific year / era
PROMPT_HH = """You are extracting specific track/era identification from a broad music conversation.

The user is looking for a SPECIFIC defining track from a particular year,
era, or cultural moment. They know exactly what significance the song has.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- direct_request: if user names an exact song.
  "Play 'Body Like A Back Road' by Sam Hunt"
  → {{"track_name": "Body Like A Back Road", "artist_name": "Sam Hunt"}}
  "What was a defining East Coast track from 1995?" → null
- artist_name: artist mentioned or confirmed.
- tag_list: era + genre + cultural significance.
  ALWAYS include the specific year/era if mentioned:
  "1995, East Coast hip-hop, defining track, golden era"
  "2017, country, summer hit, mainstream"
  Include popularity indicators: "defining", "iconic", "essential"
- target_year: specific year if mentioned. "1995", "2017", "1979"
- target_era: broader era. "mid-90s", "late 70s", "early 2000s"
- scope_after: what user wants after finding the track.
  "same_year": more defining tracks from that exact year
  "same_era": broader era exploration
  "same_artist": more from found artist
  "different_era": jump to a different time period

Output ONLY valid JSON:
{{
  "direct_request": {{...}} or null,
  "artist_name": "..." or null,
  "tag_list": "era + genre + cultural significance",
  "target_year": "specific year" or null,
  "target_era": "broader era" or null,
  "scope_after": "same_era",
  "rejected": []
}}"""


# K-HL: varied exploration within a style anchor
PROMPT_HL = """You are extracting focused style exploration keywords from a music conversation.

The user is exploring multiple tracks within a specific style, artist catalog,
or sonic territory. They have a clear stylistic anchor but want variety within it.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: anchor artist if exploring one artist's catalog.
  "The Black Keys" → stays constant while exploring their albums.
  If exploring a style across artists: null or reference artist.
  If user says "different artist": set to null.
- tag_list: the CURRENT exploration direction from LAST message.
  Keep the style anchor constant, vary the refinement:
  "gritty blues-rock, Black Keys, Brothers/El Camino era"
  → "raw, bluesy alternative, garage rock edge"
  → "similar artists, blues-infused, garage rock"
  → "Black Keys, late-night, chill, melancholic, bluesy"
  Extract from LAST message but maintain the style anchor.
- clap_keywords: sonic description of current direction.
  "raw guitar, distorted, garage production, driving rhythm"
  → "mellow blues guitar, late-night atmosphere, introspective"
- style_anchor: the CONSTANT style throughout the session.
  "gritty blues-rock", "metal from specific eras",
  "contemporary country", "classic reggae"
- current_facet: what ASPECT user is exploring now.
  "core_sound": the main characteristic sound
  "different_artist": same style, new artists
  "different_mood": same artist/style, different mood
  "different_era": same style, different time period
  "deeper_niche": more extreme/specific within the style
- wants_different_artist: true if user explicitly asks for other artists.
- rejected: what didn't fit.

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "style anchor + current refinement",
  "clap_keywords": "sonic description of current direction",
  "style_anchor": "constant style throughout session" or null,
  "current_facet": "core_sound",
  "wants_different_artist": false,
  "rejected": []
}}"""


# K-LH: finding a remembered song / sound
PROMPT_LH = """You are helping find a specific song or sound from vague memory.

The user remembers a song or a specific sonic quality but can't place it exactly.
They describe era, genre, and distinctive qualities progressively.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: confirmed or guessed artist.
  "YES! 'Hypnotize' by Biggie!" → "The Notorious B.I.G."
  Still searching: include any artist clues.
- tag_list: cumulative clues from ALL turns.
  Build progressively, dropping what was rejected:
  "90s, hip-hop, everyone knew it"
  → "90s, hip-hop, New York, East Coast"
  → "90s, East Coast hip-hop, gritty, boom-bap, not Southern"
  → "90s, East Coast, New York, boom-bap, Mobb Deep style, dark"
  ALWAYS include era and genre anchors.
- clap_keywords: how the remembered music SOUNDS.
  "90s boom-bap production, hard drums, gritty samples,
  raw East Coast flow, dark atmosphere"
- lyric_keywords: any lyrical themes mentioned.
  "about street life", "party anthem", "romantic ballad"
  Set to null if user only describes sound.
- found: true if user confirmed finding the track.
  After found: keywords shift to "more like this".
- era_anchor: the era that stays constant.
  "90s", "70s", "2010s" — user keeps coming back to this era.
- genre_anchor: the genre that stays constant.
  "East Coast hip-hop", "trip-hop", "contemporary classical"
- narrowing_history: how search narrowed across turns.
  "broad 90s hip-hop → East Coast only → gritty boom-bap → Mobb Deep style"
  Helps understand the elimination pattern.
- rejected: what was close but wrong.
  Include the REASON — this helps narrow:
  "right era but Southern not East Coast",
  "too recent, need strictly 90s",
  "good but too experimental"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "cumulative clues with era+genre anchors",
  "clap_keywords": "sonic description of remembered track",
  "lyric_keywords": "..." or null,
  "found": false,
  "era_anchor": "constant era" or null,
  "genre_anchor": "constant genre" or null,
  "narrowing_history": "how search narrowed" or null,
  "rejected": []
}}"""


# K-LL: open discovery browsing
PROMPT_LL = """You are extracting open-ended music discovery keywords.

The user is freely browsing music with no specific target.
They react to recommendations and their taste emerges organically.
Popularity and broad appeal matter here.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: any artist user has latched onto.
  If user keeps asking for more from one artist, include them.
  If openly browsing: null.
- tag_list: current direction from LAST message.
  Track the organic evolution:
  "timeless instrumental, classic movie soundtrack"
  → "contemplative, Philip Glass, contemporary classical"
  → "atmospheric, melancholic, lesser-known"
  OR: "relaxed chill beats, nostalgic" → "lo-fi hip-hop, mid-2010s"
  Extract ONLY from the LAST message.
- clap_keywords: sonic translation of current interest.
- popularity_hint: does the user want well-known or obscure tracks?
  "mainstream": "popular", "everyone knows", "classic hit"
  "underground": "lesser-known", "hidden gems", "deep cuts"
  "any": no preference stated
- browse_mode: how the user is navigating.
  "genre_locked": staying in one genre
  "genre_hopping": jumping between genres
  "artist_locked": staying with one artist
  "era_locked": staying in one time period
  "fully_open": no constraints at all
- emerging_taste: pattern forming from positive reactions.
  "likes atmospheric + melancholic",
  "prefers instrumental over vocals",
  "gravitating toward 70s funk/soul"
  Build from what user consistently says "yes" to.
- rejected: what user didn't enjoy.

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "tag_list": "current discovery direction",
  "clap_keywords": "sonic description",
  "popularity_hint": "any",
  "browse_mode": "fully_open",
  "emerging_taste": "pattern from positive reactions" or null,
  "rejected": []
}}"""


PROMPTS = {
    "HH": PROMPT_HH,
    "HL": PROMPT_HL,
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "default": PROMPT_LL,  # Unknown specificity in K -> LL (most common, 50 sessions)
}


# ── Query builders ────────────────────────────────────────────────────────────

def _build_bge_query(data: dict, specificity: str) -> str:
    """Build the query string for the BGE metadata index."""
    parts = []
    if data.get("artist_name"):
        parts.append(f"artist_name: {data['artist_name']}")
    if data.get("tag_list"):
        parts.append(f"tag_list: {data['tag_list']}")
    # K-HH: target_year as a separate tag_list line — the year may be in the metadata
    if specificity == "HH" and data.get("target_year"):
        parts.append(f"tag_list: {data['target_year']}")
    # K-LH: reinforce era/genre anchors on separate lines
    if specificity == "LH":
        if data.get("era_anchor"):
            parts.append(f"tag_list: {data['era_anchor']}")
        if data.get("genre_anchor"):
            parts.append(f"tag_list: {data['genre_anchor']}")
    return "\n".join(parts)


def _build_attr_query(data: dict) -> str:
    """Build the query for attributes-qwen3 embedding search.

    Places style_anchor (K-HL) and emerging_taste (K-LL) first
    to emphasize the current direction.
    """
    prefix_parts = []
    if data.get("style_anchor"):
        prefix_parts.append(data["style_anchor"])
    if data.get("emerging_taste"):
        prefix_parts.append(data["emerging_taste"])
    base = data.get("tag_list") or ""
    if prefix_parts:
        return f"{', '.join(prefix_parts)}, {base}" if base else ", ".join(prefix_parts)
    return base


def build_result(data: dict, specificity: str, fallback: str) -> dict:
    """Convert the parsed JSON into the format the retriever consumes."""
    bge_query     = _build_bge_query(data, specificity) or fallback
    clap_keywords = data.get("clap_keywords") or data.get("tag_list") or fallback
    attr_query    = _build_attr_query(data) or fallback

    # lyrics-qwen3: lyric_keywords can only appear in K-LH
    lyrics_query  = data.get("lyric_keywords") if specificity == "LH" else None

    # When to enable popularity:
    # HH: always (era-defining tracks should be real hits)
    # mainstream hint: always
    # LL and not underground: prefer popular tracks when "any" or unset
    # LL + underground: off, since the user wants hidden gems
    popularity_hint = data.get("popularity_hint")
    use_popularity = (
        specificity == "HH"
        or popularity_hint == "mainstream"
        or (specificity == "LL" and popularity_hint != "underground")
    )

    result = {
        "direct_request":     data.get("direct_request"),
        "bge_query":          bge_query,
        "clap_keywords":      clap_keywords,
        "attr_query":         attr_query,
        "lyrics_query":       lyrics_query,
        "rejected":           data.get("rejected") or [],
        "use_popularity":     use_popularity,
    }
    # Pass category-specific fields through unchanged
    for key in ("found", "target_year", "target_era", "scope_after",
                "style_anchor", "current_facet", "wants_different_artist",
                "era_anchor", "genre_anchor", "narrowing_history",
                "browse_mode", "popularity_hint", "emerging_taste"):
        if key in data:
            result[key] = data[key]
    return result
