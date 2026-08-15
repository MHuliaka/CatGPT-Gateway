"""
ChatGPT client — core interaction logic.

Sends messages, waits for responses, manages conversations.
Handles selector fallbacks and integrates human-like behavior.
"""

from __future__ import annotations

import asyncio
import re
import time
from urllib.parse import urlparse

from patchright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from src.config import Config
from src.selectors import Selectors
from src.browser.human import human_type, human_click, thinking_pause, random_delay
from src.chatgpt.backend_response import is_conversation_request
from src.chatgpt.detector import (
    wait_for_response_complete,
    extract_last_response_via_copy,
    count_assistant_messages,
    get_latest_assistant_turn_signature,
    is_incomplete_response_text,
)
from src.chatgpt.image_handler import extract_images_from_response
from src.chatgpt.models import ChatResponse
from src.log import setup_logging

log = setup_logging("chatgpt_client")


class ChatGPTClient:
    """
    High-level client for interacting with the ChatGPT web interface.

    Requires a Playwright Page that is already logged in and on chatgpt.com.
    """

    def __init__(self, page: Page) -> None:
        self._page = page
        self._setup_network_logging()

    def _setup_network_logging(self) -> None:
        """Monitor network requests, WebSockets, and JS errors for debugging."""
        # Only log important API calls at INFO; sentinel/ping/heartbeat at DEBUG
        _important_paths = ("/f/conversation", "/conversations?", "/stream_status")

        def on_request(request):
            url = request.url
            if "backend-api" in url or "backend-anon" in url:
                if any(p in url for p in _important_paths):
                    log.info(f"NET REQ: {request.method} {url[:200]}")
                else:
                    log.debug(f"NET REQ: {request.method} {url[:200]}")

        async def on_response(response):
            url = response.url
            if "backend-api" in url or "backend-anon" in url:
                if any(p in url for p in _important_paths):
                    log.info(f"NET RESP: {response.status} {url[:200]}")
                else:
                    log.debug(f"NET RESP: {response.status} {url[:200]}")

        def on_request_failed(request):
            url = request.url
            failure = request.failure or "unknown"
            if "chrome-extension" not in url and "favicon" not in url:
                # Patchright internal injection is expected to fail
                if "patchright" in url:
                    log.debug(f"NET FAIL: {url[:150]} — {failure}")
                else:
                    log.warning(f"NET FAIL: {url[:150]} — {failure}")

        def on_console(msg):
            if msg.type == "error":
                log.info(f"JS ERROR: {msg.text[:300]}")
            elif msg.type == "warning":
                log.debug(f"JS WARNING: {msg.text[:300]}")

        def on_page_error(error):
            log.error(f"JS PAGE ERROR: {error}")

        def on_websocket(ws):
            log.debug(f"WS OPEN: {ws.url[:200]}")
            ws.on("framereceived", lambda payload: log.debug(f"WS RECV: {str(payload)[:200]}"))
            ws.on("framesent", lambda payload: log.debug(f"WS SEND: {str(payload)[:200]}"))
            ws.on("close", lambda _: log.debug(f"WS CLOSE: {ws.url[:200]}"))

        self._page.on("request", on_request)
        self._page.on("response", on_response)
        self._page.on("requestfailed", on_request_failed)
        self._page.on("console", on_console)
        self._page.on("pageerror", on_page_error)
        self._page.on("websocket", on_websocket)

    @property
    def page(self) -> Page:
        return self._page

    # ── Core: Send & Receive ────────────────────────────────────

    async def send_message(
        self,
        text: str,
        image_paths: list[str] | None = None,
        file_paths: list[str] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """
        Send a message to ChatGPT and wait for the complete response.

        Args:
            text: The message text to send.
            image_paths: Optional list of local file paths to images to attach.
            file_paths: Optional list of local file paths to non-image files (PDF, etc.).
            model: Optional API model identifier. Browser selection is unchanged.

        Steps:
        1. Simulate thinking pause
        2. Upload images if provided
        3. Find and focus chat input
        4. Type message with human-like delays
        5. Click send
        6. Wait for the latest assistant turn to finish in the page
        7. Press Page Down and click the last visible Copy button on the page
        8. Read the copied response text from the clipboard

        Returns ChatResponse with the assistant's reply and metadata.
        """
        all_attachments = (image_paths or []) + (file_paths or [])
        log.info(f"Sending message ({len(text)} chars, {len(all_attachments)} attachments): {text[:80]}...")
        start_time = time.time()

        # 0. Check page health — recover from DNS errors before trying to send
        page_error = await self._detect_page_error()
        if page_error:
            log.warning(f"Page error detected before send: {page_error}")
            raise RuntimeError(f"Page is in error state: {page_error}")

        # Track the latest assistant turn so extraction cannot return a reply
        # from the previous request.
        pre_count = await count_assistant_messages(self._page)
        pre_turn_signature = await get_latest_assistant_turn_signature(self._page)
        log.debug(f"Assistant messages before send: {pre_count}")
        log.debug(f"Latest assistant turn before send: {pre_turn_signature}")

        # 0.5 Check for and dismiss any blocking dialogs/overlays
        await self._dismiss_overlays()

        # 1. Brief pause (human would take a moment to start typing)
        await random_delay(100, 300)

        # 1.5. Upload files/images if provided
        if all_attachments:
            await self._upload_files(all_attachments)

        # 2. Find the chat input (retry once after dismissing overlays if not found)
        input_selector = await self._find_selector(Selectors.CHAT_INPUT, "chat input")
        if not input_selector:
            # An overlay may have blocked it — dismiss and retry
            log.info("Chat input not found on first try, dismissing overlays and retrying...")
            await self._dismiss_overlays()
            await asyncio.sleep(1)
            input_selector = await self._find_selector(Selectors.CHAT_INPUT, "chat input")
        if not input_selector:
            raise RuntimeError("Could not find chat input element")

        # Arm the submission listener before text entry. The current frontend
        # can submit during insert_text(), so attaching later creates a race.
        loop = asyncio.get_running_loop()
        submission_deadline = loop.time() + (Config.RESPONSE_TIMEOUT / 1000)

        try:
            async with self._page.expect_request(
                is_conversation_request,
                timeout=Config.RESPONSE_TIMEOUT,
            ) as request_info:
                request_waiter = asyncio.ensure_future(request_info.value)

                # 3. Paste the message (all at once).
                await human_type(self._page, input_selector, text)

                # 4. Detect auto-submit from the real conversation POST. This
                # listener only detects submission; response text is copied
                # from the completed assistant turn in the page below.
                remaining = max(submission_deadline - loop.time(), 0.001)
                request_done, _ = await asyncio.wait(
                    {request_waiter}, timeout=min(3.0, remaining)
                )
                auto_submitted = bool(request_done)

                if auto_submitted:
                    request_waiter.result()
                    log.info(
                        "ChatGPT auto-submitted after text entry — "
                        "skipping send button click"
                    )
                else:
                    log.info("No auto-submit detected, clicking send button")
                    sent = await self._click_send()
                    if not sent:
                        log.info("Send button not found, trying Enter key")
                        await self._page.keyboard.press("Enter")

                remaining = max(submission_deadline - loop.time(), 0.001)
                await asyncio.wait_for(
                    asyncio.shield(request_waiter), timeout=remaining
                )
        except (asyncio.TimeoutError, PlaywrightTimeoutError) as exc:
            raise TimeoutError(
                "ChatGPT conversation request was not submitted within the configured timeout"
            ) from exc

        # Wait for the latest assistant turn to expose its completion controls.
        log.info("Waiting for ChatGPT response in the page...")
        completed = await wait_for_response_complete(
            self._page,
            expected_msg_count=pre_count + 1,
            previous_turn_signature=pre_turn_signature,
        )
        if not completed:
            log.warning("Response may not be complete (timeout)")

        # Give the completed turn a brief chance to settle before extraction.
        await asyncio.sleep(0.2)

        # Image turns may not expose a Copy button, so detect them first.
        images = await extract_images_from_response(self._page)
        has_images = len(images) > 0

        if has_images:
            response_text = await self._extract_image_turn_text(pre_turn_signature)
            log.info(f"Response contains {len(images)} generated image(s)")
            for img in images:
                log.info(f"  Image: {img.alt or img.prompt_title} → {img.local_path}")
        else:
            response_text = await extract_last_response_via_copy(
                self._page,
                previous_turn_signature=pre_turn_signature,
            )

            if not response_text.strip():
                log.warning("Empty response extracted — retrying after short wait")
                for retry in range(1, 4):
                    await asyncio.sleep(1.5 * retry)
                    response_text = await extract_last_response_via_copy(
                        self._page,
                        previous_turn_signature=pre_turn_signature,
                    )
                    if response_text.strip():
                        log.info(f"Got response on extraction retry {retry}")
                        break

            if is_incomplete_response_text(response_text):
                log.warning(
                    "Extracted text looks incomplete/transient; retrying for final answer"
                )
                for attempt in range(1, 3):
                    await asyncio.sleep(2)
                    retry_text = await extract_last_response_via_copy(
                        self._page,
                        previous_turn_signature=pre_turn_signature,
                    )

                    if retry_text and not is_incomplete_response_text(retry_text):
                        response_text = retry_text
                        log.info(f"Recovered final response text on retry {attempt}")
                        break

                    if retry_text:
                        response_text = retry_text
                    log.warning(f"Retry {attempt} still incomplete/transient")

                if is_incomplete_response_text(response_text):
                    raise TimeoutError(
                        "ChatGPT response did not complete within the configured timeout"
                    )

        elapsed_ms = int((time.time() - start_time) * 1000)
        thread_id = self._extract_thread_id()

        log.info(
            f"Response received ({elapsed_ms}ms, {len(response_text)} chars"
            f"{f', {len(images)} images' if has_images else ''}): "
            f"{response_text[:80]}..."
        )

        return ChatResponse(
            message=response_text,
            thread_id=thread_id,
            response_time_ms=elapsed_ms,
            images=images,
            has_images=has_images,
        )

    # ── Navigation ──────────────────────────────────────────────

    async def new_chat(self) -> None:
        """Start a new conversation.

        Strategy order:
        1. Immediate DOM click on ChatGPT's SPA new-chat control
        2. JavaScript navigation to the site root
        3. Short full navigation waiting only for the response to commit

        Each strategy has its own small budget. This is important because a
        regular Playwright click can otherwise spend the entire default
        30-second timeout waiting for actionability or navigation, preventing
        every fallback below it from running.
        """
        if await self._wait_for_fresh_chat(timeout_ms=1000):
            log.info("Already on a fresh chat — skipping navigation")
            return

        # Strategy 1: invoke the DOM click directly. Unlike ElementHandle.click,
        # this returns immediately and does not inherit Playwright's 30s wait.
        clicked_selector = None
        try:
            clicked_selector = await asyncio.wait_for(
                self._page.evaluate(
                    """
                    (selectors) => {
                        for (const selector of selectors) {
                            for (const element of document.querySelectorAll(selector)) {
                                const style = window.getComputedStyle(element);
                                const rect = element.getBoundingClientRect();
                                const visible = style.display !== 'none' &&
                                    style.visibility !== 'hidden' &&
                                    rect.width > 0 && rect.height > 0;
                                if (visible) {
                                    element.click();
                                    return selector;
                                }
                            }
                        }
                        return null;
                    }
                    """,
                    Selectors.NEW_CHAT_BUTTON,
                ),
                timeout=1.5,
            )
        except Exception as exc:
            log.debug(f"Fast new-chat click failed: {exc}")

        if clicked_selector:
            log.info(f"New chat via fast SPA click: {clicked_selector}")
            if await self._wait_for_fresh_chat(timeout_ms=4000):
                return
            log.warning("SPA new-chat click did not produce a ready empty chat")

        # Strategy 2: start navigation without waiting for a load event. A
        # readiness poll below decides when the new composer is actually usable.
        try:
            log.info("New chat via JS navigation...")
            await asyncio.wait_for(
                self._page.evaluate(
                    "() => window.location.assign(new URL('/', window.location.origin).href)"
                ),
                timeout=1.5,
            )
        except Exception as exc:
            # Navigation often destroys the execution context before evaluate()
            # resolves. Still poll because the navigation may have started.
            log.debug(f"JS navigation returned an error: {exc}")

        if await self._wait_for_fresh_chat(timeout_ms=5000):
            log.info("New chat started (JS navigation)")
            return

        # Strategy 3: wait only for the HTTP response to commit. Waiting for
        # DOMContentLoaded is unnecessary and is the common source of long hangs.
        try:
            log.info("New chat via short page.goto fallback...")
            await self._page.goto(
                Config.CHATGPT_URL,
                wait_until="commit",
                timeout=min(max(Config.NEW_CHAT_TIMEOUT, 1000), 7000),
            )
        except Exception as exc:
            log.warning(f"Short page.goto fallback returned an error: {exc}")

        if await self._wait_for_fresh_chat(timeout_ms=5000):
            log.info("New chat started (short page.goto)")
            return

        try:
            page_error = await asyncio.wait_for(
                self._detect_page_error(),
                timeout=1.0,
            )
        except Exception:
            page_error = None
        if page_error:
            raise RuntimeError(f"Could not start a new chat: {page_error}")
        raise RuntimeError(
            f"Provider did not open an isolated new chat (current URL: {self._page.url})"
        )

    async def _wait_for_fresh_chat(self, timeout_ms: int) -> bool:
        """Wait briefly for an empty chat with a visible composer."""
        deadline = time.monotonic() + (max(timeout_ms, 1) / 1000)
        input_selector = ", ".join(Selectors.CHAT_INPUT)

        while time.monotonic() < deadline:
            remaining_seconds = max(0.001, deadline - time.monotonic())
            try:
                fresh = await asyncio.wait_for(
                    self._is_fresh_chat(),
                    timeout=min(1.0, remaining_seconds),
                )
            except Exception:
                fresh = False

            if fresh:
                remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                try:
                    composer = await self._page.wait_for_selector(
                        input_selector,
                        timeout=min(750, remaining_ms),
                        state="visible",
                    )
                    if composer:
                        await asyncio.sleep(0.2)
                        return True
                except Exception:
                    pass

            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds > 0:
                await asyncio.sleep(min(0.2, remaining_seconds))

        return False

    async def _is_fresh_chat(self) -> bool:
        """Return true only when URL and DOM both show an empty chat."""
        current_url = urlparse(self._page.url)
        target_url = urlparse(Config.CHATGPT_URL)
        if (
            current_url.scheme not in {"http", "https"}
            or current_url.netloc != target_url.netloc
        ):
            return False

        if self._extract_thread_id():
            return False

        try:
            turn_count = await self._page.evaluate(
                """
                () => document.querySelectorAll([
                    '[data-testid^="conversation-turn-"]',
                    '[data-message-author-role="user"]',
                    '[data-message-author-role="assistant"]',
                    'section[data-turn="user"]',
                    'section[data-turn="assistant"]'
                ].join(',')).length
                """
            )
            return turn_count == 0
        except Exception:
            # An unverifiable page must never be treated as isolated.
            return False

    async def _wait_for_chat_input(self) -> None:
        """Wait for the chat input to become visible and interactive."""
        selector = ", ".join(Selectors.CHAT_INPUT)
        try:
            await self._page.wait_for_selector(
                selector,
                timeout=Config.SELECTOR_TIMEOUT,
                state="visible",
            )
            log.debug("Chat input ready")
            # Brief settle for React handlers to attach
            await asyncio.sleep(0.5)
        except Exception:
            log.warning("Chat input not found — page may not be fully ready")
            raise RuntimeError("New chat opened without an interactive chat input")

    async def _detect_page_error(self) -> str | None:
        """Check if the current page shows a browser or ChatGPT error."""
        try:
            return await self._page.evaluate(
                """
                () => {
                    const body = document.body ? document.body.innerText : '';
                    const title = document.title || '';
                    if (body.includes('DNS_PROBE_FINISHED_NXDOMAIN')) return 'DNS_PROBE_FINISHED_NXDOMAIN';
                    if (body.includes('ERR_NAME_NOT_RESOLVED')) return 'ERR_NAME_NOT_RESOLVED';
                    if (body.includes('ERR_CONNECTION_REFUSED')) return 'ERR_CONNECTION_REFUSED';
                    if (body.includes('ERR_INTERNET_DISCONNECTED')) return 'ERR_INTERNET_DISCONNECTED';
                    if (body.includes('ERR_CONNECTION_TIMED_OUT')) return 'ERR_CONNECTION_TIMED_OUT';
                    if (title.includes("can't be reached") || title.includes("is not available"))
                        return 'page_unreachable';
                    if (body.includes('Something went wrong')) return 'ChatGPT_error';
                    return null;
                }
                """
            )
        except Exception:
            return None

    async def navigate_to_thread(self, thread_id: str) -> None:
        """Navigate to an existing conversation thread."""
        url = f"{Config.CHATGPT_URL}/c/{thread_id}"
        log.info(f"Navigating to thread: {thread_id}")
        await self._page.goto(url, wait_until="domcontentloaded")
        await random_delay(800, 1500)
        log.info(f"Thread {thread_id} loaded")

    async def get_current_thread_url(self) -> str:
        """Get the current page URL (contains thread ID if in a conversation)."""
        return self._page.url

    # ── Sidebar ─────────────────────────────────────────────────

    async def list_threads(self) -> list[dict]:
        """
        Scrape the sidebar for recent conversation threads.

        Returns a list of dicts: [{id, title, url}, ...]
        """
        threads = []
        for selector in Selectors.SIDEBAR_THREAD_LINKS:
            try:
                elements = await self._page.query_selector_all(selector)
                for el in elements:
                    href = await el.get_attribute("href") or ""
                    title = (await el.inner_text()).strip()
                    match = re.search(r"/c/([a-f0-9-]+)", href)
                    if match:
                        threads.append({
                            "id": match.group(1),
                            "title": title,
                            "url": f"{Config.CHATGPT_URL}{href}",
                        })
                if threads:
                    break
            except Exception as e:
                log.debug(f"Sidebar scrape with {selector} failed: {e}")

        log.info(f"Found {len(threads)} threads in sidebar")
        return threads

    # ── Private Helpers ─────────────────────────────────────────

    async def _extract_image_turn_text(
        self,
        previous_turn_signature: str | None = None,
    ) -> str:
        """Extract descriptive text from the latest generated-image turn."""
        text = await self._page.evaluate(
            """
            (previousSignature) => {
                const turns = document.querySelectorAll(
                    'section[data-testid^="conversation-turn-"]'
                );

                for (let idx = turns.length - 1; idx >= 0; idx--) {
                    const turn = turns[idx];
                    const turnRole = turn.getAttribute('data-turn');
                    const hasAssistantRole = turnRole === 'assistant' ||
                        Boolean(turn.querySelector(
                            '[data-message-author-role="assistant"]'
                        ));
                    if (!hasAssistantRole) continue;

                    const stableId =
                        turn.getAttribute('data-turn-id') ||
                        turn.getAttribute('data-testid') ||
                        turn.id ||
                        '';
                    const signature = `${idx}:${stableId}`;
                    if (previousSignature && signature === previousSignature) {
                        return '';
                    }

                    const spans = turn.querySelectorAll('span');
                    const parts = [];
                    for (const span of spans) {
                        const value = (span.innerText || '').trim();
                        if (value && value.length > 3 && value.length < 300 &&
                            !value.includes('ChatGPT') && !value.includes('said')) {
                            parts.push(value);
                        }
                    }
                    if (parts.length > 0) return parts.join(' ');

                    return (turn.innerText || '')
                        .trim()
                        .replace(/^ChatGPT said:\\s*/i, '')
                        .trim();
                }

                return '';
            }
            """,
            previous_turn_signature,
        )
        return text or ""

    async def _find_selector(self, selectors: list[str], name: str) -> str | None:
        """
        Try each selector in the fallback list. Return the first one that matches.
        """
        for selector in selectors:
            try:
                el = await self._page.wait_for_selector(
                    selector,
                    timeout=Config.SELECTOR_TIMEOUT,
                    state="visible",
                )
                if el:
                    log.debug(f"Found {name} via: {selector}")
                    return selector
            except Exception:
                log.debug(f"Selector miss for {name}: {selector}")
                continue

        log.warning(f"No working selector found for: {name}")
        return None

    async def _dismiss_overlays(self) -> None:
        """Check for and dismiss any blocking dialogs/overlays on the page."""
        try:
            result = await self._page.evaluate(
                """
                () => {
                    const info = { dismissed: [], found: [] };

                    // Check for role="dialog" overlays
                    const dialogs = document.querySelectorAll('[role="dialog"], [role="alertdialog"], dialog[open]');
                    for (const d of dialogs) {
                        const text = (d.innerText || '').trim().substring(0, 200);
                        info.found.push('dialog: ' + text);

                        // Try to find and click dismiss/close buttons
                        const closeBtn = d.querySelector(
                            'button[aria-label="Close"], button[aria-label="Dismiss"], ' +
                            'button:has(svg[data-testid="close"]), button.close'
                        );
                        if (closeBtn) {
                            closeBtn.click();
                            info.dismissed.push('dialog-close');
                        }
                    }

                    // Check for "Continue generating" button
                    const allButtons = document.querySelectorAll('button');
                    for (const btn of allButtons) {
                        const btnText = (btn.innerText || '').trim().toLowerCase();
                        if (btnText.includes('continue generating')) {
                            btn.click();
                            info.dismissed.push('continue-generating');
                        }
                    }

                    // Check for rate limit or error banners
                    const banners = document.querySelectorAll('[class*="banner"], [class*="toast"], [class*="alert"]');
                    for (const b of banners) {
                        const text = (b.innerText || '').trim().substring(0, 200);
                        if (text) info.found.push('banner: ' + text);
                    }

                    return info;
                }
                """
            )
            if result and isinstance(result, dict):
                if result.get("dismissed"):
                    log.info(f"Dismissed overlays: {result['dismissed']}")
                if result.get("found"):
                    log.debug(f"Page overlays found: {result['found']}")
        except Exception as e:
            log.debug(f"Overlay check failed: {e}")

    async def _click_send(self) -> bool:
        """Try to click the send button using selector fallbacks."""
        # Check send button state before clicking
        btn_state = await self._page.evaluate(
            """
            () => {
                const selectors = [
                    'button[data-testid="send-button"]',
                    '#composer-submit-button',
                    "button[aria-label='Send prompt']",
                ];
                for (const sel of selectors) {
                    const btn = document.querySelector(sel);
                    if (btn) {
                        return {
                            selector: sel,
                            disabled: btn.disabled,
                            ariaDisabled: btn.getAttribute('aria-disabled'),
                            visible: btn.offsetParent !== null,
                            classes: btn.className.substring(0, 100),
                        };
                    }
                }
                return null;
            }
            """
        )
        log.debug(f"Send button state: {btn_state}")

        # Don't click a disabled send button — the input wasn't recognized
        if isinstance(btn_state, dict) and btn_state.get("disabled"):
            log.warning("Send button is disabled — text may not have been inserted properly")
            return False

        selector = await self._find_selector(Selectors.SEND_BUTTON, "send button")
        if selector:
            await human_click(self._page, selector)
            log.info(f"Send button clicked via: {selector}")
            return True
        return False

    async def _upload_files(self, file_paths: list[str]) -> None:
        """
        Upload files (images, PDFs, docs, etc.) to ChatGPT's input area.

        ChatGPT has a hidden <input type="file"> that accepts various file types.
        We set files on it directly (like drag-and-drop / file picker).
        """
        from pathlib import Path

        valid_paths = []
        for p in file_paths:
            path = Path(p)
            if path.exists() and path.is_file():
                valid_paths.append(str(path.resolve()))
            else:
                log.warning(f"File not found, skipping: {p}")

        if not valid_paths:
            log.warning("No valid files to upload")
            return

        log.info(f"Uploading {len(valid_paths)} file(s)...")

        # Find the file input element — ChatGPT has a hidden <input type="file">
        file_input = None
        for selector in Selectors.FILE_UPLOAD_INPUT:
            try:
                elements = await self._page.query_selector_all(selector)
                if elements:
                    file_input = elements[0]
                    log.debug(f"Found file input: {selector}")
                    break
            except Exception:
                continue

        if file_input:
            # Set files directly on the input element
            await file_input.set_input_files(valid_paths)
            log.info(f"Set {len(valid_paths)} file(s) on file input")
        else:
            # Fallback: use page.set_input_files with a broad selector
            log.info("No file input found via selectors, trying broad input[type=file]")
            try:
                await self._page.set_input_files("input[type='file']", valid_paths)
                log.info(f"Set {len(valid_paths)} file(s) via broad selector")
            except Exception as e:
                log.error(f"Failed to upload files: {e}")
                raise RuntimeError(f"Could not upload files: {e}")

        # Wait for files to be processed/attached (thumbnails/badges appear)
        await asyncio.sleep(3)
        # Additional wait if multiple files
        if len(valid_paths) > 1:
            await asyncio.sleep(len(valid_paths))
        log.info("File upload complete")

    def _extract_thread_id(self) -> str:
        """Extract the thread/conversation ID from the current URL."""
        url = self._page.url
        match = re.search(r"/c/([a-f0-9-]+)", url)
        return match.group(1) if match else ""
