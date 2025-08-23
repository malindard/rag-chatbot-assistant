import time
from dataclasses import dataclass
from typing import Optional
from groq import Groq
import config

@dataclass
class LLMConfig:
    model: str = getattr(config, "GROQ_MODEL")
    temperature: float = config.TEMPERATURE
    max_new_tokens: int = config.MAX_NEW_TOKENS
    max_retries: int = config.LLM_MAX_RETRIES_PER_MODEL
    backoff_seconds: float = config.LLM_BACKOFF_SECONDS

class ChatLLM:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        self.cfg = cfg or LLMConfig()
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        last_err = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_new_tokens,
                    stream=False,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                # basic diagnostics
                status = getattr(e, "status", None) or getattr(getattr(e, "respond", None), "status_code", None)
                text = getattr(getattr(e, "response", None), "text", None)
                print(f"[GROQ ERROR] attempt{attempt}/{self.cfg.max_retries} model={self.cfg.model}")
                if status: print(f"\nstatus: {status}")
                if text: print(f"\nbody: {text}")

                # retry only on 5xx or unknown
                if status in (500, 502, 503, 504) or status is None:
                    time.sleep(self.cfg.backoff_seconds * attempt)
                    continue
                raise RuntimeError(f"Groq error (ststus={status}): {text or repr(e)}")
        raise RuntimeError(f"Groq call failed after retries (model={self.cfg.model}): {repr(last_err)}")