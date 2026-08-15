"""Shared helpers for browser conversation lifecycle management."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

from src.config import Config
from src.log import setup_logging


log = setup_logging("chat_lifecycle")


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


async def return_to_home(client: Any) -> bool:
    """Return a browser provider to its fresh home page after a call.

    When a response was produced, it has already been captured at this point.
    Navigation is best-effort: a cleanup failure is logged but never replaces
    a valid response that is ready to be returned to the API caller.
    """
    if not Config.uses_browser():
        return False

    try:
        delay = max(Config.POST_RESPONSE_HOME_DELAY_SECONDS, 0.0)
        if delay:
            log.info(
                f"Provider call finished; waiting {delay:g}s before returning home"
            )
            await asyncio.sleep(delay)

        await start_new_chat(client)
    except Exception as exc:
        log.warning(f"Could not return provider browser to its home page: {exc}")
        return False

    log.info("Provider browser returned to a fresh home page")
    return True


@asynccontextmanager
async def return_home_after_call(
    client: Any,
    on_return: Callable[[], None] | None = None,
) -> AsyncIterator[None]:
    """Keep post-call home navigation inside the caller's browser lock."""
    try:
        yield
    finally:
        returned_home = await return_to_home(client)
        if returned_home and on_return is not None:
            try:
                on_return()
            except Exception as exc:
                log.warning(f"Post-home state update failed: {exc}")
