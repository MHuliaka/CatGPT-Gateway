from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.chatgpt import detector


class ChatGPTPageDownTests(unittest.TestCase):
    def _page(self) -> MagicMock:
        page = MagicMock()
        page.evaluate = AsyncMock()
        page.keyboard.press = AsyncMock()
        page.context.grant_permissions = AsyncMock()
        return page

    def test_blurs_active_element_and_presses_page_down(self) -> None:
        page = self._page()

        asyncio.run(detector._page_down_before_copy(page))

        page.evaluate.assert_awaited_once()
        page.keyboard.press.assert_awaited_once_with("PageDown")

    def test_still_presses_page_down_when_blur_fails(self) -> None:
        page = self._page()
        page.evaluate.side_effect = RuntimeError("page is still rendering")

        asyncio.run(detector._page_down_before_copy(page))

        page.keyboard.press.assert_awaited_once_with("PageDown")

    def test_copy_sequence_waits_point_eight_seconds(self) -> None:
        page = self._page()
        events: list[str] = []
        clipboard_values = iter(["old response", "new response"])

        async def evaluate(script: str, *_args):
            if "document.activeElement" in script:
                events.append("blur")
                return None
            if "navigator.clipboard.readText" in script:
                events.append("read-clipboard")
                return next(clipboard_values)
            if "navigator.clipboard.writeText" in script:
                events.append("clear-clipboard")
                return None
            if "btn.click()" in script:
                events.append("click-copy")
                return {"clicked": True, "reason": "ok", "signature": "1:new"}
            raise AssertionError(f"Unexpected evaluate script: {script[:80]}")

        async def press(key: str) -> None:
            self.assertEqual(key, "PageDown")
            events.append("page-down")

        async def sleep(delay: float) -> None:
            events.append(f"sleep-{delay}")

        page.evaluate.side_effect = evaluate
        page.keyboard.press.side_effect = press

        with patch.object(detector.asyncio, "sleep", side_effect=sleep):
            response = asyncio.run(detector.extract_last_response_via_copy(page))

        self.assertEqual(response, "new response")
        self.assertEqual(
            events,
            [
                "read-clipboard",
                "clear-clipboard",
                "blur",
                "page-down",
                "click-copy",
                "sleep-0.8",
                "read-clipboard",
            ],
        )


if __name__ == "__main__":
    unittest.main()
