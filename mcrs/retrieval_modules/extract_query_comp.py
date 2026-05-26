import json
import re
import torch

_STRUCTURED_EXTRACTION_PROMPT = """You are a music search query extractor for a conversational recommender system.

You will receive:
- [LAST USER MESSAGE]: what the user wants RIGHT NOW — this is your PRIMARY focus
- [CONVERSATION CONTEXT]: previous turns for background only

Your job: extract what the user wants to hear next, based on their LAST MESSAGE.

Output format (JSON only):
{
  "direct_request": {"track_name": "...", "artist_name": "..."} or null,
  "artist_name": "artist most relevant to the LAST user message, or null",
  "tag_list": "comma-separated mood/genre/instrument/era keywords from the LAST user message",
  "rejected": ["artists or songs explicitly rejected anywhere in the conversation"]
}

Rules:
- direct_request: ONLY when the last message explicitly names a specific song to play
- tag_list: extract from the LAST message words — mood, energy, genre, instrument, era
- artist_name: who the user wants next (from last message; ignore if they asked for a different artist)
- rejected: anything the user said NOT, "not again", "different from", "avoid" across all turns
- Do NOT invent. Only use what is stated.

Return ONLY valid JSON. No explanation or markdown."""


def extract_structured_query(conversation_str: str, lm_components=None) -> dict:
    """Extract structured search components from a conversation string.

    Focuses keyword extraction on the last user message.

    Returns:
        dict with keys:
          direct_request: {track_name, artist_name} or None
          bge_query: corpus-format string for BGE retrieval
          clap_keywords: mood/sonic keyword string for CLAP
          rejected: list of rejected artist/song names
    """
    last_user_msg = _last_user_message(conversation_str)
    fallback_query = last_user_msg
    default = {
        "direct_request": None,
        "bge_query": fallback_query,
        "clap_keywords": fallback_query,
        "rejected": [],
    } # If extraction fails, fallback to using the last user message as a generic query for bot BGE and CLAP

    if lm_components is None:
        return default

    model, tokenizer, device = lm_components
    try:
        # Explicitly separate last user message from context so Qwen knows where to focus
        user_content = (
            f"[LAST USER MESSAGE]\n{last_user_msg}\n\n"
            f"[CONVERSATION CONTEXT]\n{conversation_str}"
        ) # Qwen3 가 LAST USER MESSAGE에 집중하도록 명시적으로 구분

        messages = [
            {"role": "system", "content": _STRUCTURED_EXTRACTION_PROMPT},
            {"role": "user", "content": user_content},
            ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) # Qwen 3 formatted input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Strip <think>...</think> blocks (Qwen3 thinking mode leakage)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        # Fallback: extract first {...} block if raw has surrounding text
        if not raw.startswith("{"):
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            raw = m.group() if m else raw

        data = json.loads(raw)

        # Build BGE query in corpus field format so it matches the index
        bge_parts = []
        if data.get("artist_name"):
            bge_parts.append(f"artist_name: {data['artist_name']}")
        if data.get("tag_list"):
            bge_parts.append(f"tag_list: {data['tag_list']}")
        bge_query = "\n".join(bge_parts) if bge_parts else fallback_query

        # track_name: Blue in Green, artist_name: Miles Davis, tag_list: jazz, cool jazz, mellow

        print(f"[Qwen] direct={data.get('direct_request')} | bge='{bge_query}' | clap='{data.get('tag_list')}' | rejected={data.get('rejected', [])}")

        return {
            "direct_request": data.get("direct_request"),
            "bge_query": bge_query,
            "clap_keywords": data.get("tag_list") or fallback_query,
            "rejected": data.get("rejected") or [],
        }
    except Exception as e:
        print(f"[Qwen] extraction failed ({e}), using heuristic")
        return default


def _last_user_message(conversation_str: str) -> str:
    '''Extract the last user message from the conversation string.'''
    last = ""
    for line in conversation_str.strip().split('\n'):
        line = line.strip()
        if line.startswith("user:"):
            last = line[5:].strip()
    return last if last else conversation_str[:200]


# Kept for backwards compatibility
def extract_keyword_query(conversation_str: str, lm_components=None) -> str:
    return extract_structured_query(conversation_str, lm_components)["clap_keywords"]
