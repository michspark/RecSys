import json
import re
import torch
from .extraction_prompts import get_prompt, get_result_builder


def extract_structured_query(conversation_str: str, lm_components=None,
                              category: str = None, specificity: str = None) -> dict:
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
        # Category+specificity별 전용 프롬프트 선택 (placeholder 채우기)
        prompt_template = get_prompt(category, specificity)
        system_prompt = prompt_template.format(
            last_user_msg=last_user_msg,
            conversation_str=conversation_str,
        ) if "{last_user_msg}" in prompt_template else prompt_template

        # default.PROMPT는 placeholder가 없으므로 user_content에 직접 넣음
        if "{last_user_msg}" in prompt_template:
            user_content = ""  # 이미 system_prompt에 포함됨
        else:
            user_content = (
                f"[LAST USER MESSAGE]\n{last_user_msg}\n\n"
                f"[CONVERSATION CONTEXT]\n{conversation_str}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) # Qwen 3 formatted input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw_original = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        raw = _extract_json(raw_original)

        if not raw.strip():
            raise ValueError(f"No JSON found. Original output: {repr(raw_original[:200])}")

        data = json.loads(raw)

        # Normalize: Qwen sometimes returns list instead of comma-separated string.
        # Convert list fields to strings before passing to build_result.
        _TEXT_FIELDS = {
            "tag_list", "clap_keywords", "lyric_keywords",
            "visual_keywords", "discovered_elements", "constraints",
        }
        for field in _TEXT_FIELDS:
            if field in data and isinstance(data[field], list):
                data[field] = ", ".join(str(v) for v in data[field])

        # Build BGE query in corpus field format so it matches the index
        bge_parts = []
        if data.get("artist_name"):
            bge_parts.append(f"artist_name: {data['artist_name']}")
        if data.get("tag_list"):
            bge_parts.append(f"tag_list: {data['tag_list']}")
        bge_query = "\n".join(bge_parts) if bge_parts else fallback_query

        # track_name: Blue in Green, artist_name: Miles Davis, tag_list: jazz, cool jazz, mellow

        # 카테고리 전용 빌더가 있으면 우선 사용 (bge_query / clap_keywords 구성 방식이 다름)
        builder = get_result_builder(category)
        if builder:
            result = builder(data, specificity or "", fallback_query)
        else:
            # 범용 빌더: artist_name + tag_list → bge_query, clap_keywords 필드 우선
            bge_parts = []
            if data.get("artist_name"):
                bge_parts.append(f"artist_name: {data['artist_name']}")
            if data.get("tag_list"):
                bge_parts.append(f"tag_list: {data['tag_list']}")
            bge_query = "\n".join(bge_parts) if bge_parts else fallback_query

            clap_keywords = (
                data.get("clap_keywords")
                or data.get("tag_list")
                or fallback_query
            )
            result = {
                "direct_request": data.get("direct_request"),
                "bge_query":      bge_query,
                "clap_keywords":  clap_keywords,
                "rejected":       data.get("rejected") or [],
            }
            for extra_key in ("found", "continue_from"):
                if extra_key in data:
                    result[extra_key] = data[extra_key]

        print(f"[Qwen/{category or '-'}/{specificity or '-'}] "
              f"direct={result.get('direct_request')} | "
              f"bge='{result['bge_query'][:60]}' | "
              f"clap='{result['clap_keywords'][:60]}' | "
              f"rejected={result.get('rejected', [])}")
        return result
    except Exception as e:
        print(f"[Qwen] extraction failed ({e}), using heuristic")
        print(f"[Qwen] raw was: {repr(raw[:200])}")
        return default


def _extract_json(raw: str) -> str:
    """Raw 모델 출력에서 JSON 블록을 추출한다.

    전략 1: <think>...</think> 제거 후 JSON 탐색
    전략 2: think 제거 후 JSON이 없으면 원본(raw)에서 직접 탐색
            → JSON이 think 블록 안에 있거나 </think>가 잘린 경우 처리
    """
    def _find_json(text: str) -> str:
        """텍스트에서 첫 번째 {...} 블록을 찾아 반환."""
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        if text.startswith("{"):
            return text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return m.group() if m else ""

    # 전략 1: 완전한 <think>...</think> 블록 제거 후 JSON 탐색
    stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    result = _find_json(stripped)
    if result:
        return result

    # 전략 2: think 제거 후 JSON이 없으면 원본 전체에서 탐색
    # (JSON이 think 안에 있거나, </think>가 잘려서 제거가 안 된 경우)
    result = _find_json(raw)
    return result


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
