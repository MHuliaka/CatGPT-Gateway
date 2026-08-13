from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from src.chatgpt import detector as chatgpt_detector
from src.claude import detector as claude_detector


class ResponseTimeoutBudgetTests(unittest.TestCase):
    def test_chatgpt_fallbacks_share_one_timeout_budget(self) -> None:
        primary = AsyncMock(return_value=None)
        stop_button = AsyncMock(return_value=False)
        text_stability = AsyncMock(return_value=False)

        with patch.object(
            chatgpt_detector, "_count_copy_buttons", AsyncMock(return_value=0)
        ):
            with patch.object(
                chatgpt_detector, "_wait_for_copy_button_or_image", primary
            ):
                with patch.object(
                    chatgpt_detector, "_wait_via_stop_button", stop_button
                ):
                    with patch.object(
                        chatgpt_detector, "_wait_via_text_stability", text_stability
                    ):
                        with patch.object(
                            chatgpt_detector,
                            "monotonic",
                            side_effect=[0.0, 0.01, 0.101, 0.101],
                        ):
                            completed = asyncio.run(
                                chatgpt_detector.wait_for_response_complete(
                                    object(), timeout_ms=100
                                )
                            )

        self.assertFalse(completed)
        self.assertLessEqual(primary.await_args.args[1], 90)
        stop_button.assert_not_awaited()
        text_stability.assert_not_awaited()

    def test_claude_fallbacks_share_one_timeout_budget(self) -> None:
        streaming = AsyncMock(return_value=False)
        copy_button = AsyncMock(return_value=False)
        text_stability = AsyncMock(return_value=False)

        with patch.object(
            claude_detector, "_count_copy_buttons", AsyncMock(return_value=0)
        ):
            with patch.object(
                claude_detector, "_wait_for_streaming_complete", streaming
            ):
                with patch.object(
                    claude_detector, "_wait_for_copy_button", copy_button
                ):
                    with patch.object(
                        claude_detector, "_wait_via_text_stability", text_stability
                    ):
                        with patch.object(
                            claude_detector,
                            "monotonic",
                            side_effect=[0.0, 0.01, 0.101, 0.101],
                        ):
                            completed = asyncio.run(
                                claude_detector.wait_for_response_complete(
                                    object(), timeout_ms=100
                                )
                            )

        self.assertFalse(completed)
        self.assertLessEqual(streaming.await_args.args[1], 90)
        copy_button.assert_not_awaited()
        text_stability.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
