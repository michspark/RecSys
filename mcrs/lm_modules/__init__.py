from .llama import LLAMA_MODEL, LLAMA_MODEL_v2

def load_lm_module(lm_type, device, attn_implementation, dtype, lm_model=None):
    """Build the LM backend.

    Two paths:
      - lm_model (config block) present: it carries ALL LM settings. `version` selects the class
        ("v1" = LLAMA_MODEL, "v2" = LLAMA_MODEL_v2 / method E); model_name + attn_implementation fall
        back to the top-level lm_type/attn_implementation when omitted inside the block.
      - lm_model absent: backward-compatible behavior driven by the top-level lm_type.
    """
    # New, config-block-driven path: lm_model holds every LM knob.
    if lm_model is not None:
        version = lm_model.get("version", "v1")
        model_name = lm_model.get("model_name", lm_type)
        attn = lm_model.get("attn_implementation", attn_implementation)
        if version == "v2":
            # Method E: instruct prompt mode + sampling + natural system prompt.
            return LLAMA_MODEL_v2(
                model_name=model_name,
                device=device,
                attn_implementation=attn,
                dtype=dtype,
                prompt_mode=lm_model.get("prompt_mode", "instruct"),
                do_sample=lm_model.get("do_sample", True),
                temperature=lm_model.get("temperature", 0.8),
                top_p=lm_model.get("top_p", 0.9),
                max_new_tokens=lm_model.get("max_new_tokens", 128),
                system_prompt=lm_model.get("system_prompt", "natural"),
            )
        if version == "v1":
            # Original Llama backend, but selectable through the lm_model block too.
            return LLAMA_MODEL(model_name=model_name, device=device, attn_implementation=attn, dtype=dtype)
        raise ValueError(f"Unsupported lm_model.version: {version}")

    # Backward-compatible path: existing configs that only set the top-level lm_type.
    if lm_type == "meta-llama/Llama-3.2-1B-Instruct":
        return LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    else:
        raise ValueError(f"Unsupported LM type: {lm_type}")
