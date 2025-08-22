import time
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
import config

@dataclass
class LLMConfig:
    model: str = config.LLM_MODEL
    temperature: float = config.TEMPERATURE
    max_new_tokens: int = config.MAX_NEW_TOKENS

class ChatLLM:
    """
    OpenRouter-only chat wrapper
    """
    def __init__(self, cfg: Optional[LLMConfig] = None):
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set in .env")
        self.cfg = cfg or LLMConfig()
        self.client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request to OpenRouter via OpenAI SDK
        Includes a small retry for transient provider errors
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        attempts = 0
        last_err = None
        while attempts < 3:
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_new_tokens,
                    messages=messages,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                attempts += 1
                time.sleep(0.6 * attempts)  # 0.6s, 1.2s

        raise RuntimeError(f"LLM call failed after retries: {last_err}")
