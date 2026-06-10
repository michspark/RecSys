import os
from typing import Optional
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
    "The recommendation system has already chosen a track for you. Your only job is to introduce "
    "that track to the user in a warm, natural way, the way a friend with great music taste would.\n\n"
    "Write 2-3 sentences that:\n"
    "- connect the track to something specific the user said earlier in the conversation\n"
    "- describe what it sounds or feels like in plain, everyday language\n"
    "- read like a real person talking, not a chatbot\n\n"
    "Do not gush (\"I'm so glad!\", \"perfect!\"), do not list metadata fields, and do not end by asking "
    "whether they want more recommendations. Just talk."
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
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
        system_prompt: str = "natural",
    ) -> None:
        # Store generation/prompt knobs before loading (super().__init__ loads the model).
        self.prompt_mode = prompt_mode
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
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