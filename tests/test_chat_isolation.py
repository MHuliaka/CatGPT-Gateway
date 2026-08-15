from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch

from src.api import openai_routes
from src.api.chat_lifecycle import NewChatTimeoutError, start_new_chat
from src.api.openai_schemas import ChatCompletionRequest, ChatMessage
from src.chatgpt.models import ChatResponse
from src.config import Config


class RecordingBrowserClient:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.send_count = 0

    async def new_chat(self) -> None:
        self.events.append("new_chat")

    async def send_message(
        self,
        text: str,
        image_paths=None,
        file_paths=None,
        model: str | None = None,
        stateless: bool = False,
    ) -> ChatResponse:
        self.events.append("send_message")
        self.send_count += 1
        return ChatResponse(
            message=f"response {self.send_count}",
            thread_id=f"thread-{self.send_count}",
        )


class HomeResetFailingBrowserClient(RecordingBrowserClient):
    async def new_chat(self) -> None:
        self.events.append("new_chat")
        raise RuntimeError("navigation failed")


class SlowBrowserClient:
    async def new_chat(self) -> None:
        await asyncio.sleep(1)


class ChatIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_client = openai_routes._client
        self.previous_lock = openai_routes._lock
        self.previous_response_time = openai_routes._last_response_time
        self.previous_thread_count = openai_routes._thread_message_count
        openai_routes._lock = None
        openai_routes._last_response_time = 0.0
        openai_routes._thread_message_count = 0

    def tearDown(self) -> None:
        openai_routes._client = self.previous_client
        openai_routes._lock = self.previous_lock
        openai_routes._last_response_time = self.previous_response_time
        openai_routes._thread_message_count = self.previous_thread_count

    def test_each_chat_completion_returns_browser_home_after_response(self) -> None:
        client = RecordingBrowserClient()
        openai_routes._client = client
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")]
        )

        async def run_requests() -> None:
            await openai_routes.create_chat_completion(request)
            await openai_routes.create_chat_completion(request)

        with patch.object(Config, "PROVIDER", "chatgpt"):
            with patch.object(Config, "POST_RESPONSE_HOME_DELAY_SECONDS", 0):
                with patch.object(Config, "NEW_CHAT_TIMEOUT", 1000):
                    with patch.object(openai_routes, "_MIN_MESSAGE_GAP", 0):
                        asyncio.run(run_requests())

        self.assertEqual(
            client.events,
            ["send_message", "new_chat", "send_message", "new_chat"],
        )
        self.assertEqual(openai_routes._thread_message_count, 0)

    def test_home_navigation_failure_does_not_replace_captured_response(self) -> None:
        client = HomeResetFailingBrowserClient()
        openai_routes._client = client
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")]
        )

        with patch.object(Config, "PROVIDER", "chatgpt"):
            with patch.object(Config, "POST_RESPONSE_HOME_DELAY_SECONDS", 0):
                with patch.object(openai_routes, "_MIN_MESSAGE_GAP", 0):
                    response = asyncio.run(
                        openai_routes.create_chat_completion(request)
                    )

        self.assertEqual(response.choices[0].message.content, "response 1")
        self.assertEqual(client.events, ["send_message", "new_chat"])

    def test_new_chat_navigation_has_one_total_timeout(self) -> None:
        started = time.monotonic()
        with patch.object(Config, "NEW_CHAT_TIMEOUT", 10):
            with self.assertRaises(NewChatTimeoutError):
                asyncio.run(start_new_chat(SlowBrowserClient()))

        self.assertLess(time.monotonic() - started, 0.5)


if __name__ == "__main__":
    unittest.main()
