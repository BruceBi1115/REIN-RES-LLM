from __future__ import annotations

import os
from typing import Any


class OpenAINewsApiAdapter:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.1",
        base_url: str | None = None,
        timeout_sec: float = 30.0,
        max_retries: int = 2,
        live_logger: Any = None,
    ):
        from openai import OpenAI

        cfg: dict[str, Any] = {"api_key": str(api_key).strip()}
        if str(base_url or "").strip():
            cfg["base_url"] = str(base_url).strip()
        self.client = OpenAI(**cfg)
        self.model = str(model or "gpt-5.1").strip() or "gpt-5.1"
        self.timeout_sec = float(max(1.0, timeout_sec))
        self.max_retries = int(max(0, max_retries))
        self.live_logger = live_logger

    @staticmethod
    def _extract_text(resp) -> str:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        parts: list[str] = []
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            for item in output:
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content")
                if not isinstance(content, list):
                    continue
                for chunk in content:
                    if isinstance(chunk, dict):
                        text = chunk.get("text") or chunk.get("output_text")
                    else:
                        text = getattr(chunk, "text", None) or getattr(chunk, "output_text", None)
                    if text:
                        parts.append(str(text))
        return "\n".join(parts).strip()

    def chat_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        sys_text = str(system_prompt)
        usr_text = str(user_prompt)
        token_cap = int(max(1, max_tokens))
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": usr_text},
        ]
        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                if hasattr(self.client, "responses"):
                    resp = self.client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "system", "content": [{"type": "input_text", "text": sys_text}]},
                            {"role": "user", "content": [{"type": "input_text", "text": usr_text}]},
                        ],
                        max_output_tokens=token_cap,
                        timeout=self.timeout_sec,
                    )
                    out = self._extract_text(resp)
                    if out:
                        return out
            except Exception as exc:
                last_error = exc

            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=token_cap,
                    timeout=self.timeout_sec,
                )
                if getattr(resp, "choices", None):
                    content = resp.choices[0].message.content
                    if isinstance(content, list):
                        content = "".join(str(x) for x in content)
                    text = str(content or "").strip()
                    if text:
                        return text
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        return ""


def _read_api_key_from_path(path: str) -> str:
    p = str(path or "").strip()
    if not p or not os.path.exists(p):
        return ""
    with open(p, "r", encoding="utf-8") as handle:
        return str(handle.read()).strip()


def discover_news_api_key(args) -> tuple[str, str]:
    env_key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    if env_key:
        return env_key, "env"

    requested_path = str(getattr(args, "news_api_key_path", "") or "").strip()
    candidate_paths = []
    if requested_path:
        candidate_paths.append(requested_path)
    for fallback_path in [".secrets/api_key.txt", "api_key.txt"]:
        if fallback_path not in candidate_paths:
            candidate_paths.append(fallback_path)

    for path in candidate_paths:
        file_key = _read_api_key_from_path(path)
        if file_key:
            return file_key, path
    return "", ""


def build_news_api_adapter(args, live_logger=None):
    api_key, source = discover_news_api_key(args)
    if not api_key:
        return None
    if live_logger is not None:
        live_logger.info(
            f"[NEWS_API] adapter enabled with model={getattr(args, 'news_api_model', 'gpt-5.1')} key_source={source or 'unknown'}"
        )
    return OpenAINewsApiAdapter(
        api_key=api_key,
        model=str(getattr(args, "news_api_model", "gpt-5.1") or "gpt-5.1"),
        base_url=str(getattr(args, "news_api_base_url", "") or "").strip() or None,
        timeout_sec=float(getattr(args, "news_api_timeout_sec", 30.0) or 30.0),
        max_retries=int(getattr(args, "news_api_max_retries", 2) or 2),
        live_logger=live_logger,
    )
