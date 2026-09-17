from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import openai_routes
from src.chatgpt.models import ChatResponse, ImageInfo
from src.config import Config


class RecordingImageClient:
    def __init__(self, generated_path: str) -> None:
        self.generated_path = generated_path
        self.image_inputs: list[tuple[str, bytes]] = []
        self.file_inputs: list[tuple[str, bytes]] = []

    async def new_chat(self) -> None:
        return None

    async def send_message(
        self,
        text: str,
        image_paths: list[str] | None = None,
        file_paths: list[str] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        self.image_inputs = [
            (Path(path).name, Path(path).read_bytes()) for path in image_paths or []
        ]
        self.file_inputs = [
            (Path(path).name, Path(path).read_bytes()) for path in file_paths or []
        ]
        return ChatResponse(
            message="",
            images=[
                ImageInfo(
                    url="https://example.test/generated.png",
                    local_path=self.generated_path,
                )
            ],
            has_images=True,
        )


class ImageGenerationInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_client = openai_routes._client
        self.previous_lock = openai_routes._lock
        self.previous_response_time = openai_routes._last_response_time
        self.previous_thread_count = openai_routes._thread_message_count
        openai_routes._lock = None
        openai_routes._last_response_time = 0.0
        openai_routes._thread_message_count = 0

        app = FastAPI()
        app.include_router(openai_routes.openai_router)
        self.http = TestClient(app)

    def tearDown(self) -> None:
        self.http.close()
        openai_routes._client = self.previous_client
        openai_routes._lock = self.previous_lock
        openai_routes._last_response_time = self.previous_response_time
        openai_routes._thread_message_count = self.previous_thread_count

    def test_json_request_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "generated.png"
            generated.write_bytes(b"generated")
            client = RecordingImageClient(str(generated))
            openai_routes._client = client

            with patch.object(Config, "PROVIDER", "chatgpt"), patch.object(
                Config, "POST_RESPONSE_HOME_DELAY_SECONDS", 0
            ), patch.object(openai_routes, "_MIN_MESSAGE_GAP", 0):
                response = self.http.post(
                    "/v1/images/generations",
                    json={"prompt": "A cat in space", "response_format": "b64_json"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["data"][0]["b64_json"],
            base64.b64encode(b"generated").decode("ascii"),
        )
        self.assertEqual(client.image_inputs, [])
        self.assertEqual(client.file_inputs, [])

    def test_multipart_images_and_files_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "generated.png"
            generated.write_bytes(b"generated")
            client = RecordingImageClient(str(generated))
            openai_routes._client = client

            with patch.object(Config, "PROVIDER", "chatgpt"), patch.object(
                Config, "POST_RESPONSE_HOME_DELAY_SECONDS", 0
            ), patch.object(openai_routes, "_MIN_MESSAGE_GAP", 0):
                response = self.http.post(
                    "/v1/images/generations",
                    data={"prompt": "Use both inputs", "response_format": "b64_json"},
                    files=[
                        ("input_file", ("reference.png", b"image bytes", "image/png")),
                        ("input_file", ("brief.txt", b"file bytes", "text/plain")),
                    ],
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(client.image_inputs), 1)
        self.assertTrue(client.image_inputs[0][0].endswith("_reference.png"))
        self.assertEqual(client.image_inputs[0][1], b"image bytes")
        self.assertEqual(len(client.file_inputs), 1)
        self.assertTrue(client.file_inputs[0][0].endswith("_brief.txt"))
        self.assertEqual(client.file_inputs[0][1], b"file bytes")


if __name__ == "__main__":
    unittest.main()
