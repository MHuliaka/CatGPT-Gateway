from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from src.chatgpt.client import ChatGPTClient


class ChatGPTPageDownTests(unittest.TestCase):
    def _client(self) -> tuple[ChatGPTClient, MagicMock]:
        page = MagicMock()
        page.evaluate = AsyncMock()
        page.keyboard.press = AsyncMock()
        return ChatGPTClient(page), page

    def test_blurs_active_element_and_presses_page_down(self) -> None:
        client, page = self._client()

        asyncio.run(client._page_down_before_response())

        page.evaluate.assert_awaited_once()
        page.keyboard.press.assert_awaited_once_with("PageDown")

    def test_still_presses_page_down_when_blur_fails(self) -> None:
        client, page = self._client()
        page.evaluate.side_effect = RuntimeError("page is still rendering")

        asyncio.run(client._page_down_before_response())

        page.keyboard.press.assert_awaited_once_with("PageDown")

    def test_page_down_failure_does_not_fail_response_flow(self) -> None:
        client, page = self._client()
        page.keyboard.press.side_effect = RuntimeError("keyboard unavailable")

        asyncio.run(client._page_down_before_response())

        page.keyboard.press.assert_awaited_once_with("PageDown")


if __name__ == "__main__":
    unittest.main()
