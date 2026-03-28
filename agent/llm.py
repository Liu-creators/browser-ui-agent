"""LLM Client

Supports OpenAI and Anthropic vision models for UI-Agent reasoning.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic.

    Automatically selects backend based on model name prefix:
    - Models starting with 'gpt' or 'o1' -> OpenAI
    - Models starting with 'claude' -> Anthropic
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = self._build_client(api_key)

    def _build_client(self, api_key: Optional[str]):
        """Instantiate the appropriate backend client."""
        if self._is_openai():
            try:
                from openai import AsyncOpenAI
                key = api_key or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise ValueError("OPENAI_API_KEY not set")
                return AsyncOpenAI(api_key=key)
            except ImportError as exc:
                raise ImportError("Install openai: pip install openai") from exc

        elif self._is_anthropic():
            try:
                import anthropic
                key = api_key or os.environ.get("ANTHROPIC_API_KEY")
                if not key:
                    raise ValueError("ANTHROPIC_API_KEY not set")
                return anthropic.AsyncAnthropic(api_key=key)
            except ImportError as exc:
                raise ImportError("Install anthropic: pip install anthropic") from exc

        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _is_openai(self) -> bool:
        return any(self.model.startswith(p) for p in ("gpt", "o1", "o3"))

    def _is_anthropic(self) -> bool:
        return self.model.startswith("claude")

    async def vision_chat(
        self,
        system: str,
        user_text: str,
        image_b64: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> str:
        """Send a vision-capable chat message.

        Args:
            system: System prompt.
            user_text: User message text.
            image_b64: Optional base64-encoded PNG image.
            response_format: Set to 'json' to request JSON output.

        Returns:
            Model response as string.
        """
        if self._is_openai():
            return await self._openai_chat(system, user_text, image_b64, response_format)
        return await self._anthropic_chat(system, user_text, image_b64)

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    async def _openai_chat(
        self,
        system: str,
        user_text: str,
        image_b64: Optional[str],
        response_format: Optional[str],
    ) -> str:
        messages = [{"role": "system", "content": system}]

        user_content: list = [{"type": "text", "text": user_text}]
        if image_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": "high",
                },
            })
        messages.append({"role": "user", "content": user_content})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Anthropic backend
    # ------------------------------------------------------------------

    async def _anthropic_chat(
        self,
        system: str,
        user_text: str,
        image_b64: Optional[str],
    ) -> str:
        user_content: list = []
        if image_b64:
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            })
        user_content.append({"type": "text", "text": user_text})

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        block = response.content[0] if response.content else None
        return block.text if block and hasattr(block, "text") else ""

    async def chat(self, system: str, user_text: str) -> str:
        """Text-only chat (no image)."""
        return await self.vision_chat(system=system, user_text=user_text)
