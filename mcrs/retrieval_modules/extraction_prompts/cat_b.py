"""Category B extraction prompts — finding tracks by lyrics / story."""

# B-HH: finding a song from an exact lyric line
PROMPT_HH = """You are extracting lyrics-focused search keywords from a music conversation.

The user is looking for a song by its EXACT LYRICS or very specific narrative content.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: if the user already found the song or mentions an artist.
  After turn 1, the user often explores more from the same artist.
  If they say "Can you recommend another Green Day song", artist_name = "Green Day".
- lyric_keywords: the EXACT quoted lyrics or key narrative phrases from the
  user's description. Keep the user's original wording as close as possible.
  Example: "Do you have the time to listen to me whine"
  If no exact lyrics in the LAST message, extract the lyrical THEME
  the user is now asking about (e.g., "punk-rock energy, early 2000s emo").
- tag_list: genre, style, and mood descriptors mentioned.
  Example: "punk rock, pop-punk, alternative rock, energetic, fast tempo"
- phase: "searching" if still looking for the original song,
  "found" if they confirmed finding it (said "YES", "that's the one", "perfect"),
  "exploring" if they found it and are now asking for similar songs.
- rejected: artists, songs, or styles the user explicitly rejected.
  Include "too pop", "too heavy", "not what I'm looking for" context.

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "lyric_keywords": "exact lyrics or lyrical theme",
  "tag_list": "genre, style, mood descriptors",
  "phase": "searching",
  "rejected": []
}}"""


# B-HL: exploring several songs around a specific artist
PROMPT_HL = """You are extracting artist-focused search keywords from a music conversation.

The user is deep-diving into a SPECIFIC ARTIST's discography,
exploring different songs by them or with specific thematic focus.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: the PRIMARY artist the user is exploring. This is the
  MOST IMPORTANT field. Even if user says "or similar artists",
  keep the main artist here.
  Example: conversation about Anitta → artist_name = "Anitta"
- tag_list: what ASPECT of the artist the user wants now.
  This changes each turn as the user explores different facets:
  "female empowerment, assertive" → "upbeat, dancing" → "international collabs"
  → "relaxed, chill, ballad"
  Extract ONLY from the LAST message, not the whole conversation.
- lyric_keywords: thematic or lyrical focus if mentioned.
  Example: "storytelling about struggle", "narrative about a journey",
  "songs about rebellion or defiance"
  Set to null if user is asking by sound/vibe rather than lyrics.
- wants_different_artist: true if user says "different artist",
  "other bands", "not just [artist]". false otherwise.
  If true, tag_list becomes more important than artist_name.
- rejected: specific songs already played that user didn't like,
  or styles they said don't fit.

Output ONLY valid JSON:
{{
  "artist_name": "primary artist being explored",
  "tag_list": "current turn's thematic/style focus",
  "lyric_keywords": "lyrical theme" or null,
  "wants_different_artist": false,
  "rejected": []
}}"""


# B-LH: finding a song remembered by its lyric content / story
PROMPT_LH = """You are helping find a song the user remembers by its LYRICAL CONTENT or STORY.

The user has a song in mind but remembers it by what the lyrics are ABOUT,
not the exact words. They are narrowing down through thematic clues.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: any artist the user mentions or confirms finding.
  If they say "YES! DUCKWORTH by Kendrick!", artist_name = "Kendrick Lamar".
  If still searching, include artist guesses if any: "maybe Kendrick or Kevin Gates"
- lyric_keywords: the STORY or THEME the user describes.
  Focus on narrative elements, not sound:
  "personal story about a turning point"
  "family history, specific event, crazy storytelling"
  "vivid storytelling, watching a movie, super clear details"
  Use the user's LAST message words as much as possible.
- tag_list: genre + any stylistic clues.
  Example: "hip-hop, storytelling, conscious rap, dense lyrics"
- found: true if user confirmed ("YES!", "that's the one!", "you found it").
  After found, lyric_keywords should shift to describe what they want NEXT.
- rejected: tracks that were close but not right.
  Include WHY they were rejected if the user explained:
  "FEAR was good but too general, not about a specific event"

Output ONLY valid JSON:
{{
  "artist_name": "..." or null,
  "lyric_keywords": "story/theme description from LAST message",
  "tag_list": "genre and style descriptors",
  "found": false,
  "rejected": []
}}"""


# B-LL: broad artist exploration (discovery, not a specific song)
PROMPT_LL = """You are extracting broad artist exploration keywords from a music conversation.

The user is casually exploring an artist's discography or discovering
songs within a general theme. No specific track in mind.

[LAST USER MESSAGE]
{last_user_msg}

[CONVERSATION CONTEXT]
{conversation_str}

Extract:
- artist_name: the artist being explored. If user is now asking for
  "other artists" or "different bands", keep the REFERENCE artist
  but note wants_different_artist = true.
  Example: exploring Weezer → artist_name = "Weezer"
- tag_list: what the user wants from the LAST message.
  This evolves each turn:
  "interesting lyrics, thought-provoking" → "humorous theme"
  → "introspective, melancholic" → "high-energy, rock out"
  → "anthemic, singalong" → "quirky, unique sound"
  Extract ONLY from the LAST message.
- lyric_keywords: lyrical theme if the user mentions it.
  "thought-provoking lyrics" → "personal narratives, humor"
  Set to null if user is asking about sound/vibe, not lyrics.
- wants_different_artist: true if user explicitly asks for other artists.
- mood_shift: describe how the user's request changed from the previous turn.
  "from reflective to high-energy", "from lyrics-focus to sound-focus"
  This helps select the right facet of the artist.

Output ONLY valid JSON:
{{
  "artist_name": "artist being explored",
  "tag_list": "current turn's style/vibe focus",
  "lyric_keywords": "lyrical theme" or null,
  "wants_different_artist": false,
  "mood_shift": "..." or null,
  "rejected": []
}}"""


PROMPTS = {
    "HH": PROMPT_HH,
    "HL": PROMPT_HL,
    "LH": PROMPT_LH,
    "LL": PROMPT_LL,
    "default": PROMPT_LH,
}


# ── BGE / lyrics query builders ───────────────────────────────────────────────

def _build_bge_query(data: dict, specificity: str) -> str:
    """Build the query string for the BGE metadata index."""
    parts = []
    if specificity == "HH":
        # Lyrics are the key signal — put lyric_keywords in the tag_list slot
        if data.get("artist_name"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("lyric_keywords"):
            parts.append(f"tag_list: {data['lyric_keywords']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")
    elif specificity == "HL":
        # Artist is the key signal — drop the artist if wants_different_artist
        if data.get("artist_name") and not data.get("wants_different_artist"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")
    elif specificity == "LH":
        # Mix lyric themes with artist clues
        if data.get("artist_name"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("lyric_keywords"):
            parts.append(f"tag_list: {data['lyric_keywords']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")
    else:  # LL
        if data.get("artist_name") and not data.get("wants_different_artist"):
            parts.append(f"artist_name: {data['artist_name']}")
        if data.get("tag_list"):
            parts.append(f"tag_list: {data['tag_list']}")
    return "\n".join(parts)


def _build_lyrics_query(data: dict, specificity: str) -> str:
    """Build the query string for lyrics-qwen3 embedding search."""
    if specificity == "HH":
        # Exact lyric line as-is
        return data.get("lyric_keywords") or data.get("tag_list") or ""
    elif specificity == "LH":
        # Story / theme description
        return data.get("lyric_keywords") or data.get("tag_list") or ""
    else:  # HL, LL — lyrics matter less
        return data.get("lyric_keywords") or data.get("tag_list") or ""


def build_result(data: dict, specificity: str, fallback: str) -> dict:
    """Convert the parsed JSON into the format the retriever consumes."""
    bge_query    = _build_bge_query(data, specificity) or fallback
    lyrics_query = _build_lyrics_query(data, specificity) or fallback
    result = {
        "direct_request": data.get("direct_request"),
        "bge_query":      bge_query,
        "clap_keywords":  lyrics_query,   # in category B, clap_keywords = lyrics query
        "rejected":       data.get("rejected") or [],
    }
    # Pass category-specific fields through unchanged
    for key in ("phase", "found", "wants_different_artist", "lyric_keywords", "mood_shift"):
        if key in data:
            result[key] = data[key]
    return result
