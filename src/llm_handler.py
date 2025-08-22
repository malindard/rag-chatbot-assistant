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
    def __init__(self, cfg: Optional[LLMConfig] = None):
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set in .env")
        self.cfg = cfg or LLMConfig()
        self.client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        attempts, last_err = 0, None
        while attempts < 2:
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
                # --- added debug ---
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                body = getattr(getattr(e, "response", None), "text", None)
                print(f"[LLM ERROR] attempt {attempts}/2 on model={self.cfg.model}")
                if status: print(f"  status: {status}")
                if body:   print(f"  body: {body}")
                print(f"  exception: {repr(e)}")
                # --------------
                time.sleep(0.6 * attempts)

        raise RuntimeError(f"LLM call failed after retries (model={self.cfg.model}): {last_err}")
