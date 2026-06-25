import os
from typing import Optional, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLAMA_MODEL:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda", attn_implementation="eager", dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.lm, self.tokenizer = self._load_model()
        self.lm.eval()
        self.lm.to(self.device).to(self.dtype)

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        lm = AutoModelForCausalLM.from_pretrained(self.model_name, attn_implementation=self.attn_implementation, dtype=self.dtype)
        return lm, tokenizer

    def _format_chat_history(self, sys_prompt, chat_history: list, recommend_item: str):
        chat_data = [{"role": "system", "content": sys_prompt}]
        chat_data += chat_history
        chat_data += [{"role": "assistant", "content": recommend_item}]
        chat_template = self.tokenizer.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True)
        return chat_template

    def response_generation(self, sys_prompt: str, chat_history: list, recommend_item: str,max_new_tokens=512, response_format=None):
        chat_history = self._format_chat_history(sys_prompt, chat_history, recommend_item)
        token_inputs = self.tokenizer(chat_history, return_tensors="pt")
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.lm.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        generated_text = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=True)[0]
        return generated_text

    def batch_response_generation(self, sys_prompts: list[str], chat_histories: list[list], recommend_items: list[str], max_new_tokens=64):
        """Generate responses for multiple inputs in batch.

        Args:
            sys_prompts: List of system prompts.
            chat_histories: List of chat history lists.
            recommend_items: List of recommended items.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            List of generated response texts.
        """
        # Format all chat histories
        formatted_chats = [
            self._format_chat_history(sys_prompt, chat_history, recommend_item)
            for sys_prompt, chat_history, recommend_item in zip(sys_prompts, chat_histories, recommend_items)
        ]

        # Tokenize with padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        token_inputs = self.tokenizer(formatted_chats, return_tensors="pt", padding=True, truncation=True)
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.lm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode only the newly generated tokens
        generated_texts = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=True)
        return generated_texts

# ================================================================================================
# Method E instruction + system-prompt constants (self-contained inside this module).
# ================================================================================================

# "instruct" prompt mode: the retrieved track is placed inside an explicit user instruction asking
# for an explanation, instead of being pre-filled as a partial assistant turn ("inject" mode).
INSTRUCT_TEMPLATE_V2 = (
    "Here is the track our system retrieved for the user:\n{recommend_item}\n\n"
    "Write a short, personalized explanation (2-3 sentences) for why this track fits what the user "
    "is looking for, grounded in the conversation above. Speak naturally; do not list raw metadata fields."
)

# "natural" system prompt: this REPLACES the checklist response_generation block (read from
# system_prompts/response_generation.txt) inside the incoming system prompt. Kept in English
# because the conversations and the judge are English.
NATURAL_RESPONSE_PROMPT = (
    "A track has been chosen for the user. Introduce it the way a friend with great "
    "music taste would.\n\n"
    "Write 2-3 sentences that connect this track to what the user is looking for, and "
    "explain why it fits in a natural, specific way.\n\n"
    "Just talk like a real person — no gushing, no listing fields, no asking if they want more."
)

NATURAL_RESPONSE_PROMPT_GPT = (
    "The recommendation system has already chosen a track for you. Introduce it to the user "
    "the way a friend with great music taste would.\n\n"
    "Write 2-3 sentences that:\n"
    "- tie the track to something specific the user said earlier\n"
    "- name something concrete about THIS track — the artist's style, an instrument, the tempo, "
    "the era, or the lyrical theme — not generic mood words\n"
    "- sound like a real person, not a chatbot\n\n"
    "If the track doesn't fully match what the user asked for, acknowledge it honestly and say what "
    "it does offer instead.\n\n"
    "Vary your openings across responses. Avoid stock phrases like 'this track', 'perfect for', "
    "'you mentioned'.\n\n"
    "Don't gush, don't list metadata fields, don't ask if they want more. Just talk."
)

class LLAMA_MODEL_v2(LLAMA_MODEL):
    """Method E generation backend.

    Differs from LLAMA_MODEL in three ways (config-driven via the `lm_model` block):
      1. prompt_mode="instruct" — the recommend_item goes into a user instruction, not a pre-filled
         assistant turn.
      2. sampling decoding — do_sample=True with temperature/top_p (nucleus sampling).
      3. system_prompt="natural" — the checklist response_generation block in the incoming system
         prompt is swapped for NATURAL_RESPONSE_PROMPT (the checklist text is read from the same
         system_prompts/response_generation.txt that CRS_BASELINE concatenates).

    Works with any HF causal LM (e.g. Qwen/Qwen3-8B); Qwen needs enable_thinking=False in its chat
    template, which is applied automatically when the model name contains "qwen".
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        dtype: torch.dtype = torch.bfloat16,
        prompt_mode: str = "instruct",
        do_sample: bool = True,
        temperature: float = 0.95,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        max_new_tokens: int = 128,
        system_prompt: str = "natural",
    ) -> None:
        # Store generation/prompt knobs before loading (super().__init__ loads the model).
        self.prompt_mode = prompt_mode
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.max_new_tokens = max_new_tokens
        self.system_prompt_mode = system_prompt
        # Qwen chat template supports enable_thinking; Llama does not. Detect from the model name.
        self.is_qwen = "qwen" in model_name.lower()

        # Reuse LLAMA_MODEL's loading (left-padded tokenizer + eval + to(device/dtype)).
        super().__init__(
            model_name=model_name, device=device,
            attn_implementation=attn_implementation, dtype=dtype,
        )

        # Ensure a pad token exists for left-padded batch generation (Qwen may lack one).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Read the checklist response_generation block so the "natural" swap knows what to replace.
        # llama.py lives in mcrs/lm_modules/, so system_prompts is one directory up.
        prompts_dir = os.path.join(os.path.dirname(__file__), "..", "system_prompts")
        checklist_path = os.path.join(prompts_dir, "response_generation.txt")
        with open(checklist_path, "r", encoding="utf-8") as f:
            self._checklist_block = f.read()

    def _rewrite_system_prompt(self, sys_prompt: str) -> str:
        """In 'natural' mode, replace the checklist response_generation block with the natural one."""
        # Only swap when natural mode is on AND the checklist block is actually present in sys_prompt.
        if self.system_prompt_mode == "natural" and self._checklist_block in sys_prompt:
            return sys_prompt.replace(self._checklist_block, NATURAL_RESPONSE_PROMPT)
        return sys_prompt

    def _apply_template(self, messages: list) -> str:
        """apply_chat_template wrapper; passes enable_thinking=False only for Qwen models."""
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.is_qwen:
            template_kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    def _format_chat_history(self, sys_prompt: str, chat_history: list, recommend_item: str) -> str:
        """Build the templated string under the chosen prompt_mode (instruct or inject)."""
        # Apply the natural-prompt swap (no-op in checklist mode) before assembling messages.
        sys_prompt = self._rewrite_system_prompt(sys_prompt)

        if self.prompt_mode == "instruct":
            # recommend_item embedded in an explicit user instruction asking for an explanation.
            instruction = INSTRUCT_TEMPLATE_V2.format(recommend_item=recommend_item)
            messages = (
                [{"role": "system", "content": sys_prompt}]
                + chat_history
                + [{"role": "user", "content": instruction}]
            )
        elif self.prompt_mode == "inject":
            # recommend_item pre-filled as an assistant turn (same layout as LLAMA_MODEL).
            messages = (
                [{"role": "system", "content": sys_prompt}]
                + chat_history
                + [{"role": "assistant", "content": recommend_item}]
            )
        else:
            raise ValueError(f"unknown prompt_mode: {self.prompt_mode}")

        return self._apply_template(messages)

    def _generate_kwargs(self, max_new_tokens: Optional[int]) -> dict:
        """Assemble generate() kwargs: greedy vs nucleus sampling, plus pad token + length."""
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens if max_new_tokens is None else max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            # repetition_penalty > 1.0 down-weights tokens that already appeared in the sequence,
            # discouraging verbatim repeats. 1.0 = no penalty (HF default).
            "repetition_penalty": self.repetition_penalty,
            # no_repeat_ngram_size = N hard-forbids any N-gram from occurring twice in one generation.
            # 0 = disabled (HF default).
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }
        if self.do_sample:
            # temperature sharpens/flattens the next-token distribution; top_p keeps the smallest set
            # of tokens whose cumulative probability >= top_p (nucleus sampling).
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
        return generate_kwargs

    def response_generation(
        self,
        sys_prompt: str,
        chat_history: list,
        recommend_item: str,
        max_new_tokens: Optional[int] = None,
        response_format=None,
    ) -> str:
        """Single-input generation (same signature CRS_BASELINE.chat expects)."""
        formatted_chat = self._format_chat_history(sys_prompt, chat_history, recommend_item)
        token_inputs = self.tokenizer(formatted_chat, return_tensors="pt")
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.lm.generate(
                input_ids, attention_mask=attention_mask, **self._generate_kwargs(max_new_tokens)
            )
        # Decode only the newly generated tokens (strip the prompt prefix).
        generated_text = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        return generated_text

    def batch_response_generation(
        self,
        sys_prompts: list[str],
        chat_histories: list[list],
        recommend_items: list[str],
        max_new_tokens: Optional[int] = None,
    ) -> list[str]:
        """Batched generation (same signature CRS_BASELINE.batch_chat expects)."""
        # Render every prompt under the chosen mode (+ natural swap).
        formatted_chats = [
            self._format_chat_history(sys_prompt, chat_history, recommend_item)
            for sys_prompt, chat_history, recommend_item in zip(sys_prompts, chat_histories, recommend_items)
        ]

        # Tokenize with left padding (tokenizer was created with padding_side="left").
        token_inputs = self.tokenizer(formatted_chats, return_tensors="pt", padding=True, truncation=True)
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.lm.generate(
                input_ids, attention_mask=attention_mask, **self._generate_kwargs(max_new_tokens)
            )

        # Decode only the newly generated tokens for every sample.
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1]:], skip_special_tokens=True
        )
        return generated_texts

    def simple_generate(
        self,
        messages: list,
        temperature: float = 0.3,
        max_new_tokens: int = 128,
    ) -> str:
        """Single-prompt generation for the reranker (one user message, custom temperature).

        Mirrors response_generation but takes a raw messages list instead of the
        (sys_prompt, chat_history, recommend_item) triple, and forces its own temperature
        so reranking stays consistent (low) regardless of the response-generation knobs.
        """
        # Render the messages into the model's chat format (Qwen enable_thinking=False auto).
        text = self._apply_template(messages)
        token_inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)
        # Build generate kwargs locally: temperature>0 -> sampling, temperature==0 -> greedy.
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = self.top_p
        with torch.no_grad():
            outputs = self.lm.generate(
                input_ids, attention_mask=attention_mask, **generate_kwargs
            )
        # Decode only the newly generated tokens (strip the prompt prefix).
        return self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]

    def score_yes_no_batch(self, prompts: list[str], chunk_size: int = 8) -> list[float]:
        """Pointwise relevance score = p(Yes) / (p(Yes) + p(No)) for each prompt.

        Unlike simple_generate (which samples text), this does a SINGLE forward pass and reads the
        next-token distribution: the probability the model would answer "Yes" vs "No". This is the
        pointwise reranking signal — a calibrated relevance in [0, 1], no text decoding, no hallucination.

        prompts must already be chat-templated strings whose last position is right before the answer
        token (use _apply_template with add_generation_prompt=True). Returns one p_yes per prompt.
        """
        # Lazily build & cache the Yes/No token-id sets. We cover capital/leading-space/lowercase
        # variants and take the max over each set, because the exact surface form depends on context.
        if not hasattr(self, "_yes_ids"):
            yes_variants = ["Yes", " Yes", "yes"]
            no_variants = ["No", " No", "no"]
            self._yes_ids = sorted({self.tokenizer.encode(v, add_special_tokens=False)[0] for v in yes_variants})
            self._no_ids = sorted({self.tokenizer.encode(v, add_special_tokens=False)[0] for v in no_variants})

        scores: list[float] = []
        # Process in chunks so the batched forward stays within memory.
        for start in range(0, len(prompts), chunk_size):
            chunk = prompts[start: start + chunk_size]
            # Left-padded tokenizer -> real tokens are right-aligned, so column -1 is the final
            # real token for every row in the batch (the position whose logits predict the answer).
            token_inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            input_ids = token_inputs.input_ids.to(self.device)
            attention_mask = token_inputs.attention_mask.to(self.device)
            with torch.no_grad():
                # logits_to_keep=1: compute the LM head for ONLY the last position, not all T.
                # Vocab is ~152k, so full-sequence logits [B, T, V] would be multiple GB; this
                # keeps just [B, 1, V] and avoids OOM.
                logits = self.lm(
                    input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1
                ).logits  # [B, 1, V]
            last_logits = logits[:, -1, :]                                       # [B, V] next-token logits
            yes_logit = last_logits[:, self._yes_ids].max(dim=-1).values         # [B] best "Yes" variant
            no_logit = last_logits[:, self._no_ids].max(dim=-1).values           # [B] best "No" variant
            # 2-way softmax over just (Yes, No) -> p_yes = exp(yes)/(exp(yes)+exp(no)).
            pair = torch.stack([yes_logit, no_logit], dim=-1)                    # [B, 2]
            p_yes = torch.softmax(pair, dim=-1)[:, 0]                            # [B]
            scores.extend(p_yes.float().tolist())
        return scores


class OPENAI_MODEL:
    """Response-generation backend that calls the OpenAI Chat Completions API.

    Drop-in replacement for LLAMA_MODEL_v2's response generation: it exposes the same
    response_generation / batch_response_generation methods CRS_BASELINE calls, but talks to a
    remote model instead of loading a local one. No GPU weights, no tokenizer, no device/dtype.

    Reuses the method-E prompt assembly: the 'natural' system-prompt swap and INSTRUCT_TEMPLATE_V2.
    It does NOT implement score_yes_no_batch/simple_generate (those need raw logit access), so any
    config using this backend must keep the LLM reranker disabled.
    """

    def __init__(
        self,
        model_name: str = "gpt-5.4",
        temperature: float = 0.8,
        max_new_tokens: int = 128,
        system_prompt: str = "natural",
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> None:
        # Imported lazily so the openai dependency is only required for API runs.
        from openai import OpenAI

        # OpenAI() reads the API key from the OPENAI_API_KEY environment variable by default.
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_prompt_mode = system_prompt
        # OpenAI's analogues of repetition control (the HF generate knobs don't exist here):
        # frequency_penalty scales with how often a token already appeared; presence_penalty is a
        # flat penalty once a token has appeared at all. 0.0 = no penalty for both.
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # Read the checklist response_generation block so the 'natural' swap knows what to replace.
        # (Same source LLAMA_MODEL_v2 uses; llama.py lives in mcrs/lm_modules/.)
        prompts_dir = os.path.join(os.path.dirname(__file__), "..", "system_prompts")
        checklist_path = os.path.join(prompts_dir, "response_generation.txt")
        with open(checklist_path, "r", encoding="utf-8") as f:
            self._checklist_block = f.read()

    def _rewrite_system_prompt(self, sys_prompt: str) -> str:
        """In 'natural' mode, replace the checklist response_generation block with the natural one."""
        if self.system_prompt_mode == "natural" and self._checklist_block in sys_prompt:
            return sys_prompt.replace(self._checklist_block, NATURAL_RESPONSE_PROMPT)
        return sys_prompt

    def _build_messages(
        self, sys_prompt: str, chat_history: List[Dict], recommend_item: str
    ) -> List[Dict]:
        """Assemble the OpenAI messages list: system + chat history + instruct turn.

        Mirrors LLAMA_MODEL_v2's 'instruct' prompt_mode, but returns a raw messages list because the
        API consumes messages directly (no chat template, no enable_thinking).
        """
        sys_prompt = self._rewrite_system_prompt(sys_prompt)
        instruction = INSTRUCT_TEMPLATE_V2.format(recommend_item=recommend_item)
        return (
            [{"role": "system", "content": sys_prompt}]
            + chat_history
            + [{"role": "user", "content": instruction}]
        )

    def response_generation(self, sys_prompt, chat_history, recommend_item, max_new_tokens=None, response_format=None):
        messages = self._build_messages(sys_prompt, chat_history, recommend_item)
        completion = self.client.chat.completions.create(
            model=self.model_name,                          # "gpt-5.4"
            messages=messages,
            max_completion_tokens=512,                      # max_tokens 아님! reasoning 토큰 포함이라 넉넉히
            reasoning_effort="none",                        # 응답생성엔 reasoning 불필요 (none 또는 low)
            # frequency_penalty: 이미 등장한 토큰을 빈도에 비례해 down-weight → 반복 단어 억제 → LexDiv↑.
            # 그동안 생성자에서 받기만 하고 호출엔 안 넣어 무시됐던 값을 실제로 전달한다 (config로 제어).
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            # temperature는 계속 미전달 (penalty 쪽이 품질을 덜 해쳐 우선).
        )
        return completion.choices[0].message.content

    def batch_response_generation(
        self,
        sys_prompts: List[str],
        chat_histories: List[List[Dict]],
        recommend_items: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Batched generation interface. The Chat API has no native batch, so we call once per item."""
        responses: List[str] = []
        for sys_prompt, chat_history, recommend_item in zip(sys_prompts, chat_histories, recommend_items):
            responses.append(
                self.response_generation(sys_prompt, chat_history, recommend_item, max_new_tokens)
            )
        return responses