from __future__ import annotations

import asyncio
import json
import unittest

from src.chatgpt.backend_response import (
    BackendResponseError,
    is_conversation_request,
    parse_conversation_sse,
    read_conversation_response,
)


def _sse(*payloads: object) -> str:
    frames = []
    for payload in payloads:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        frames.append(f"event: delta\ndata: {data}\n\n")
    return "".join(frames)


def _message(
    text: str,
    *,
    message_id: str = "message-1",
    status: str = "finished_successfully",
    content_type: str = "text",
    recipient: str = "all",
    metadata: dict | None = None,
) -> dict:
    return {
        "id": message_id,
        "author": {"role": "assistant"},
        "recipient": recipient,
        "content": {"content_type": content_type, "parts": [text]},
        "status": status,
        "end_turn": status == "finished_successfully",
        "metadata": metadata or {},
    }


class _Request:
    def __init__(self, url: str, method: str = "POST") -> None:
        self.url = url
        self.method = method


class _Response:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body.encode()

    async def body(self) -> bytes:
        return self._body


class ConversationRequestMatcherTests(unittest.TestCase):
    def test_matches_authenticated_and_anonymous_conversation_paths(self) -> None:
        for path in (
            "/backend-api/f/conversation",
            "/backend-anon/f/conversation",
            "/backend-api/conversation",
        ):
            with self.subTest(path=path):
                self.assertTrue(
                    is_conversation_request(
                        _Request(f"https://chatgpt.com{path}?ignored=true")
                    )
                )

    def test_rejects_prepare_get_and_unrelated_requests(self) -> None:
        self.assertFalse(
            is_conversation_request(
                _Request("https://chatgpt.com/backend-anon/f/conversation/prepare")
            )
        )
        self.assertFalse(
            is_conversation_request(
                _Request(
                    "https://chatgpt.com/backend-anon/f/conversation",
                    method="GET",
                )
            )
        )
        self.assertFalse(is_conversation_request(object()))


class ConversationSseParserTests(unittest.TestCase):
    def test_uses_last_full_message_snapshot(self) -> None:
        stream = _sse(
            {
                "message": _message("Hel", status="in_progress"),
                "conversation_id": "conversation-1",
            },
            {
                "message": _message("Hello\nworld"),
                "conversation_id": "conversation-1",
            },
            "[DONE]",
        )

        result = parse_conversation_sse(stream)

        self.assertEqual(result.text, "Hello\nworld")
        self.assertEqual(result.conversation_id, "conversation-1")
        self.assertEqual(result.message_id, "message-1")

    def test_reconstructs_v1_append_and_batched_patch_events(self) -> None:
        stream = _sse(
            {
                "o": "add",
                "p": "",
                "v": {
                    "message": _message("", status="in_progress"),
                    "conversation_id": "conversation-v1",
                },
            },
            {
                "o": "append",
                "p": "/message/content/parts/0",
                "v": "Hello\n",
            },
            {
                "o": "patch",
                "p": "",
                "v": [
                    {
                        "o": "append",
                        "p": "/message/content/parts/0",
                        "v": "from backend",
                    },
                    {
                        "o": "replace",
                        "p": "/message/status",
                        "v": "finished_successfully",
                    },
                    {"o": "replace", "p": "/message/end_turn", "v": True},
                ],
            },
            {"conversation_id": "conversation-v1", "message_id": "message-1"},
            "[DONE]",
        )

        result = parse_conversation_sse(stream)

        self.assertEqual(result.text, "Hello\nfrom backend")
        self.assertEqual(result.conversation_id, "conversation-v1")
        self.assertEqual(result.message_id, "message-1")

    def test_accepts_v1_content_operations_without_root_snapshot(self) -> None:
        # Some captures begin after the root/add event. The content path itself
        # still unambiguously identifies the current conversation message.
        stream = _sse(
            {
                "o": "append",
                "p": "/message/content/parts/0",
                "v": "Hello",
            },
            {
                "o": "append",
                "p": "/message/content/parts/0",
                "v": " world",
            },
            "[DONE]",
        )

        result = parse_conversation_sse(stream)

        self.assertEqual(result.text, "Hello world")

    def test_handles_recorders_that_remove_blank_lines_between_events(self) -> None:
        first = json.dumps(
            {
                "o": "append",
                "p": "/message/content/parts/0",
                "v": "one",
            }
        )
        second = json.dumps(
            {
                "o": "append",
                "p": "/message/content/parts/0",
                "v": " two",
            }
        )
        stream = f"data: {first}\ndata: {second}\ndata: [DONE]\n"

        result = parse_conversation_sse(stream)

        self.assertEqual(result.text, "one two")

    def test_accepts_early_v1_bare_value_chunks(self) -> None:
        result = parse_conversation_sse(
            _sse({"v": "bare"}, {"v": " value"}, "[DONE]")
        )

        self.assertEqual(result.text, "bare value")

    def test_ignores_hidden_reasoning_and_tool_recipient_messages(self) -> None:
        stream = _sse(
            {
                "message": _message(
                    "private reasoning",
                    message_id="reasoning",
                    content_type="reasoning_recap",
                )
            },
            {
                "message": _message(
                    "tool instructions",
                    message_id="tool-call",
                    recipient="browser",
                )
            },
            {"message": _message("Visible answer", message_id="answer")},
            "[DONE]",
        )

        result = parse_conversation_sse(stream)

        self.assertEqual(result.text, "Visible answer")
        self.assertEqual(result.message_id, "answer")

    def test_returns_empty_text_for_a_visible_non_text_image_message(self) -> None:
        image_message = _message("", content_type="multimodal_text")
        image_message["content"]["parts"] = [
            {"content_type": "image_asset_pointer", "asset_pointer": "file-service://1"}
        ]

        result = parse_conversation_sse(_sse({"message": image_message}, "[DONE]"))

        self.assertEqual(result.text, "")
        self.assertEqual(result.content_type, "multimodal_text")

    def test_raises_for_backend_error_event(self) -> None:
        with self.assertRaisesRegex(BackendResponseError, "capacity unavailable"):
            parse_conversation_sse(
                _sse({"error": {"message": "capacity unavailable"}}, "[DONE]")
            )

    def test_raises_for_stream_without_visible_assistant_message(self) -> None:
        user_message = _message("not an answer")
        user_message["author"]["role"] = "user"

        with self.assertRaisesRegex(BackendResponseError, "no visible assistant"):
            parse_conversation_sse(_sse({"message": user_message}, "[DONE]"))


class ConversationHttpResponseTests(unittest.TestCase):
    def test_reads_successful_sse_body(self) -> None:
        response = _Response(
            200,
            _sse(
                {
                    "message": _message("direct body"),
                    "conversation_id": "conversation-http",
                },
                "[DONE]",
            ),
        )

        result = asyncio.run(read_conversation_response(response))

        self.assertEqual(result.text, "direct body")
        self.assertEqual(result.conversation_id, "conversation-http")

    def test_surfaces_non_success_backend_detail(self) -> None:
        response = _Response(403, '{"detail":"requirements token expired"}')

        with self.assertRaisesRegex(
            BackendResponseError, "HTTP 403: requirements token expired"
        ):
            asyncio.run(read_conversation_response(response))


if __name__ == "__main__":
    unittest.main()
