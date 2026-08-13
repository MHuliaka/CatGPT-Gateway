"""Shared helpers for browser conversation lifecycle management."""

from __future__ import annotations

import asyncio
from typing import Any

from src.config import Config


class NewChatTimeoutError(TimeoutError):
    """Raised when the provider UI cannot open a fresh chat in time."""


async def start_new_chat(client: Any) -> None:
    """Open a fresh provider chat within the configured time budget.

    Browser navigation can otherwise inherit Playwright's timeout for every
    selector and fallback attempt, turning one reset into a several-minute
    operation.  Keep one total budget here so every API route has the same
    bounded behavior.
    """
    timeout_seconds = max(Config.NEW_CHAT_TIMEOUT, 1) / 1000

    try:
        await asyncio.wait_for(client.new_chat(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise NewChatTimeoutError(
            f"Timed out starting a new chat after {timeout_seconds:g}s"
        ) from exc
