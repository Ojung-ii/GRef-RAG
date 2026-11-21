"""OpenAI-compatible generation and Qwen token accounting."""

from __future__ import annotations

import os
import time
from typing import Any

QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class TokenCounter:
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        revision = QWEN_REVISION if name == QWEN_MODEL else None
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)

    def context(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def full_input(self, prompt: str) -> int:
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(tokens)


class Generator:
    def __init__(self, config: dict[str, Any]):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=os.environ.get("VLLM_API_KEY") or "EMPTY",
            timeout=config["timeout_s"],
            max_retries=2,
        )

    def verify_model(self) -> None:
        model = self.config["model"]
        if model not in {entry.id for entry in self.client.models.list().data}:
            raise RuntimeError(f"The configured endpoint does not serve {model}")

    def complete(self, prompt: str, seed: int | None) -> dict[str, Any]:
        cfg = self.config
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_new_tokens"],
            **({"seed": seed} if seed is not None else {}),
        )
        elapsed = time.perf_counter() - started
        prediction = response.choices[0].message.content or ""
        return {
            "raw_prediction": prediction,
            "prediction": prediction.strip(),
            "generation_seconds": elapsed,
            "full_input_tokens": response.usage.prompt_tokens if response.usage else None,
            "output_tokens": response.usage.completion_tokens if response.usage else None,
        }
