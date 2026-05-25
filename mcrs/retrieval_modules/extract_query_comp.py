import torch

_KEYWORD_EXTRACTION_PROMPT = """You are a music search keyword extractor.
Given a music conversation, extract the most important musical keywords that describe what the user CURRENTLY wants.
Focus on: mood, energy, genre, instruments, similar artists, sonic qualities.
Ignore track IDs, filler words, and already-recommended artists the user rejected.
Return ONLY a short comma-separated list of keywords. No explanation.

Example:
User says: "I want something intense and dramatic but NOT Alesana, different artist, post-hardcore"
Output: intense, dramatic, post-hardcore, new artist

Now extract keywords from the conversation below:"""


def extract_keyword_query(conversation_str: str, lm_components=None) -> str:
    """Extract musical keywords from the conversation for CLAP retrieval.

    Args:
        conversation_str: Full conversation string (role: content format).
        lm_components: Optional tuple of (model, tokenizer, device) for a raw HF model.
                       If None, falls back to the last user message heuristic.

    Returns:
        A short keyword string describing what the user wants musically.
    """
    if lm_components is not None:
        model, tokenizer, device = lm_components

        try:
            messages = [
                {"role": "system", "content": "You are a music search keyword extractor. Reply with only a comma-separated list of keywords."},
                {"role": "user", "content": f"{_KEYWORD_EXTRACTION_PROMPT}\n{conversation_str}\nKeywords:"},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    repetition_penalty=1.2,
                )
            keywords = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            if keywords and "track_id" not in keywords and "track_name" not in keywords:
                return keywords
        except Exception:
            pass  # fall through to heuristic

    # Heuristic fallback: use only the last user message (most recent intent)
    last_user_line = ""
    for line in conversation_str.strip().split('\n'):
        line = line.strip()
        if line.startswith("user:"):
            last_user_line = line[5:].strip()
    return last_user_line if last_user_line else conversation_str[:200]
