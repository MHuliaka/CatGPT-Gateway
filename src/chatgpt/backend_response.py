"""Capture and decode ChatGPT's conversation backend response.

The browser remains responsible for authentication, Sentinel/Cloudflare tokens,
and submitting the request.  This module reads the matching conversation HTTP
response and reconstructs the assistant message from its SSE payload instead of
copying rendered text from the page.

ChatGPT currently emits either full message snapshots or a compact ``v1``
JSON-patch-like stream.  The endpoint is private, so the parser deliberately
supports both formats and only depends on the stable message/content paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

from patchright.async_api import Request, Response


_CONVERSATION_PATHS = frozenset(
    {
        "/backend-api/f/conversation",
        "/backend-anon/f/conversation",
        # Kept for older ChatGPT frontend builds.
        "/backend-api/conversation",
    }
)

_HIDDEN_CONTENT_TYPES = frozenset(
    {
        "computer_initialize_state",
        "computer_output",
        "execution_output",
        "model_editable_context",
        "reasoning_recap",
        "thoughts",
        "tool_result",
    }
)


class BackendResponseError(RuntimeError):
    """The conversation backend response could not be used."""


@dataclass(frozen=True)
class BackendConversationResponse:
    """Assistant result reconstructed from the conversation event stream."""

    text: str
    conversation_id: str = ""
    message_id: str = ""
    content_type: str = ""


@dataclass(frozen=True)
class _MessageCandidate:
    text: str
    message_id: str
    content_type: str
    order: int


def is_conversation_request(request: Request) -> bool:
    """Return whether *request* is a ChatGPT conversation submission."""

    try:
        method = request.method.upper()
        path = urlparse(request.url).path.rstrip("/")
    except (AttributeError, TypeError, ValueError):
        return False
    return method == "POST" and path in _CONVERSATION_PATHS


async def read_conversation_response(response: Response) -> BackendConversationResponse:
    """Read a Patchright response body and decode its conversation SSE events."""

    try:
        raw_body = await response.body()
    except Exception as exc:
        raise BackendResponseError(
            f"Could not read the ChatGPT backend response body: {exc}"
        ) from exc

    body = raw_body.decode("utf-8-sig", errors="replace")
    status = int(getattr(response, "status", 0) or 0)
    if status < 200 or status >= 300:
        detail = _error_detail(body)
        suffix = f": {detail}" if detail else ""
        raise BackendResponseError(
            f"ChatGPT conversation backend returned HTTP {status}{suffix}"
        )

    return parse_conversation_sse(body)


def parse_conversation_sse(stream_text: str) -> BackendConversationResponse:
    """Reconstruct the final visible assistant message from an SSE body.

    Supported input shapes:

    * classic full-message snapshots (``{"message": ...}``)
    * v1 root/add/append/replace operations
    * v1 batched ``patch`` operations
    """

    document: Any = {}
    candidates: dict[str, _MessageCandidate] = {}
    anonymous_generation = 0
    order = 0
    conversation_id = ""
    terminal_message_id = ""
    parsed_events = 0
    malformed_events = 0
    saw_assistant_message = False
    saw_message_content_operation = False

    for payload_text in _iter_sse_payloads(stream_text):
        if payload_text.strip() == "[DONE]":
            break

        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            # Comments/keepalives and rollout-specific non-JSON events do not
            # contain message text.  Count JSON-looking failures so an entirely
            # malformed stream produces a useful error below.
            if payload_text.lstrip().startswith(("{", "[")):
                malformed_events += 1
            continue

        if not isinstance(payload, dict):
            continue

        parsed_events += 1
        _raise_event_error(payload)

        found_conversation_id = _find_identifier(payload, "conversation_id")
        if found_conversation_id:
            conversation_id = found_conversation_id
        found_message_id = _find_identifier(payload, "message_id")
        if found_message_id:
            terminal_message_id = found_message_id

        # The old encoding sends a complete message on every event.
        for message in _message_dicts(payload):
            order += 1
            candidate = _candidate_from_message(message, order)
            if candidate is not None:
                saw_assistant_message = True
                _store_candidate(candidates, candidate, anonymous_generation)

        operation_payload = payload
        operation = payload.get("o")
        if not isinstance(operation, str) and isinstance(payload.get("v"), str):
            # Early v1 rollouts also emitted bare value chunks.
            operation = "append"
            operation_payload = {
                "o": "append",
                "p": "/message/content/parts/0",
                "v": payload["v"],
            }
        if isinstance(operation, str):
            if _operation_replaces_root(operation_payload):
                anonymous_generation += 1
                saw_message_content_operation = False
            if _operation_mentions_message_content(operation_payload):
                saw_message_content_operation = True
            document = _apply_operation(document, operation_payload)

            message = document.get("message") if isinstance(document, dict) else None
            if isinstance(message, dict):
                order += 1
                candidate = _candidate_from_message(
                    message,
                    order,
                    assume_assistant=saw_message_content_operation,
                )
                if candidate is not None:
                    saw_assistant_message = True
                    _store_candidate(candidates, candidate, anonymous_generation)

            if isinstance(document, dict):
                doc_conversation_id = _as_identifier(document.get("conversation_id"))
                if doc_conversation_id:
                    conversation_id = doc_conversation_id

    if candidates:
        # In a stream containing tool/reasoning messages and a final answer,
        # the last visible assistant message is the user-facing result.
        nonempty = [candidate for candidate in candidates.values() if candidate.text]
        chosen = max(nonempty or list(candidates.values()), key=lambda item: item.order)
        return BackendConversationResponse(
            text=chosen.text,
            conversation_id=conversation_id,
            message_id=chosen.message_id or terminal_message_id,
            content_type=chosen.content_type,
        )

    if saw_assistant_message:
        return BackendConversationResponse(
            text="",
            conversation_id=conversation_id,
            message_id=terminal_message_id,
        )

    if parsed_events == 0:
        if malformed_events:
            raise BackendResponseError(
                "ChatGPT backend returned an SSE stream with malformed JSON events"
            )
        detail = _error_detail(stream_text)
        suffix = f": {detail}" if detail else ""
        raise BackendResponseError(
            f"ChatGPT backend response contained no conversation events{suffix}"
        )

    raise BackendResponseError(
        "ChatGPT backend stream contained no visible assistant message"
    )


def _iter_sse_payloads(stream_text: str) -> Iterable[str]:
    """Yield data payloads from SSE, with NDJSON/plain-JSON fallbacks."""

    normalized = stream_text.replace("\r\n", "\n").replace("\r", "\n")
    pending_data: list[str] = []
    saw_sse_field = False

    for line in normalized.split("\n"):
        if line == "":
            if pending_data:
                yield "\n".join(pending_data)
                pending_data.clear()
            continue

        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if separator and field == "data":
            saw_sse_field = True
            if value.startswith(" "):
                value = value[1:]

            # ChatGPT normally separates events with a blank line.  Some
            # recorders collapse those blank lines, leaving consecutive,
            # independently valid data fields.
            if pending_data and _is_complete_json_payload("\n".join(pending_data)):
                yield "\n".join(pending_data)
                pending_data.clear()
            pending_data.append(value)
            continue

        # event/id/retry fields do not carry response content.
        if separator and field in {"event", "id", "retry"}:
            continue

        if not saw_sse_field and line.strip():
            # Non-SSE JSON errors and newline-delimited captures.
            yield line.strip()

    if pending_data:
        yield "\n".join(pending_data)


def _is_complete_json_payload(value: str) -> bool:
    if value.strip() == "[DONE]":
        return True
    try:
        json.loads(value)
        return True
    except json.JSONDecodeError:
        return False


def _operation_replaces_root(operation: dict[str, Any]) -> bool:
    return (
        operation.get("o") in {"add", "replace"}
        and operation.get("p", "") in {"", "/"}
        and isinstance(operation.get("v"), dict)
    )


def _operation_mentions_message_content(
    operation: dict[str, Any], base_path: str = ""
) -> bool:
    path = _join_pointer(base_path, operation.get("p", ""))
    if path.startswith("/message/content/parts"):
        return True
    if operation.get("o") == "patch" and isinstance(operation.get("v"), list):
        return any(
            _operation_mentions_message_content(child, path)
            for child in operation["v"]
            if isinstance(child, dict)
        )
    return False


def _apply_operation(document: Any, operation: dict[str, Any], base_path: str = "") -> Any:
    op = operation.get("o")
    path = _join_pointer(base_path, operation.get("p", ""))
    value = operation.get("v")

    if op == "patch" and isinstance(value, list):
        for child in value:
            if isinstance(child, dict):
                document = _apply_operation(document, child, path)
        return document

    if op not in {"add", "append", "remove", "replace"}:
        return document

    tokens = _pointer_tokens(path)
    if not tokens:
        if op == "remove":
            return {}
        if op == "append":
            return _append_value(document, value)
        return value

    if not isinstance(document, (dict, list)):
        document = [] if tokens[0].isdigit() else {}

    parent = document
    for index, token in enumerate(tokens[:-1]):
        next_token = tokens[index + 1]
        parent = _ensure_child(parent, token, next_token)

    final_token = tokens[-1]
    if op == "remove":
        _remove_child(parent, final_token)
    elif op == "append":
        current = _get_child(parent, final_token)
        _set_child(parent, final_token, _append_value(current, value), add=False)
    else:
        _set_child(parent, final_token, value, add=(op == "add"))
    return document


def _join_pointer(base_path: str, child_path: Any) -> str:
    base = base_path if isinstance(base_path, str) else ""
    child = child_path if isinstance(child_path, str) else ""
    if not base or base == "/":
        return child
    if not child or child == "/":
        return base
    return f"{base.rstrip('/')}/{child.lstrip('/')}"


def _pointer_tokens(path: str) -> list[str]:
    if path in {"", "/"}:
        return []
    return [
        token.replace("~1", "/").replace("~0", "~")
        for token in path.lstrip("/").split("/")
    ]


def _ensure_child(parent: Any, token: str, next_token: str) -> Any:
    create_list = next_token.isdigit() or next_token == "-"
    default: Any = [] if create_list else {}

    if isinstance(parent, dict):
        child = parent.get(token)
        if not isinstance(child, (dict, list)):
            child = default
            parent[token] = child
        return child

    if isinstance(parent, list):
        index = len(parent) if token == "-" else int(token)
        while len(parent) <= index:
            parent.append(None)
        child = parent[index]
        if not isinstance(child, (dict, list)):
            child = default
            parent[index] = child
        return child

    raise TypeError("JSON pointer parent is not a container")


def _get_child(parent: Any, token: str) -> Any:
    if isinstance(parent, dict):
        return parent.get(token)
    if isinstance(parent, list):
        if token == "-":
            return None
        index = int(token)
        return parent[index] if 0 <= index < len(parent) else None
    return None


def _set_child(parent: Any, token: str, value: Any, *, add: bool) -> None:
    if isinstance(parent, dict):
        parent[token] = value
        return
    if isinstance(parent, list):
        if token == "-":
            parent.append(value)
            return
        index = int(token)
        if add and 0 <= index < len(parent):
            parent.insert(index, value)
            return
        while len(parent) <= index:
            parent.append(None)
        parent[index] = value


def _remove_child(parent: Any, token: str) -> None:
    if isinstance(parent, dict):
        parent.pop(token, None)
    elif isinstance(parent, list) and token != "-":
        index = int(token)
        if 0 <= index < len(parent):
            parent.pop(index)


def _append_value(current: Any, value: Any) -> Any:
    if current is None:
        return value
    if isinstance(current, str):
        return current + (value if isinstance(value, str) else str(value))
    if isinstance(current, list):
        if isinstance(value, list):
            current.extend(value)
        else:
            current.append(value)
        return current
    if isinstance(current, dict) and isinstance(value, dict):
        current.update(value)
        return current
    return value


def _message_dicts(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    direct = payload.get("message")
    if isinstance(direct, dict):
        yield direct

    value = payload.get("v")
    if isinstance(value, dict):
        nested = value.get("message")
        if isinstance(nested, dict):
            yield nested

    if "author" in payload and "content" in payload:
        yield payload


def _candidate_from_message(
    message: dict[str, Any],
    order: int,
    *,
    assume_assistant: bool = False,
) -> _MessageCandidate | None:
    author = message.get("author")
    if not isinstance(author, dict) or author.get("role") != "assistant":
        if not assume_assistant or (
            isinstance(author, dict) and author.get("role") not in (None, "")
        ):
            return None

    recipient = message.get("recipient")
    if recipient not in (None, "", "all", "user"):
        return None

    metadata = message.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if any(
        metadata.get(key) is True
        for key in (
            "is_hidden",
            "hide_from_ui",
            "is_visually_hidden_from_conversation",
        )
    ):
        return None
    if metadata.get("ui_visibility") == "hidden":
        return None

    content = message.get("content")
    if not isinstance(content, dict):
        return None
    content_type = str(content.get("content_type") or "")
    if content_type in _HIDDEN_CONTENT_TYPES:
        return None

    text = _content_text(content)
    return _MessageCandidate(
        text=text,
        message_id=_as_identifier(message.get("id")),
        content_type=content_type,
        order=order,
    )


def _content_text(content: dict[str, Any]) -> str:
    parts = content.get("parts")
    if isinstance(parts, list):
        return "".join(_part_text(part) for part in parts)

    text = content.get("text")
    if isinstance(text, str):
        return text
    if isinstance(text, dict) and isinstance(text.get("value"), str):
        return text["value"]

    output_text = content.get("output_text")
    return output_text if isinstance(output_text, str) else ""


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    for key in ("text", "content", "value"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    return ""


def _store_candidate(
    candidates: dict[str, _MessageCandidate],
    candidate: _MessageCandidate,
    anonymous_generation: int,
) -> None:
    key = candidate.message_id or f"anonymous-{anonymous_generation}"
    candidates[key] = candidate


def _find_identifier(value: Any, key: str) -> str:
    if not isinstance(value, dict):
        return ""
    direct = _as_identifier(value.get(key))
    if direct:
        return direct
    nested = value.get("v")
    if isinstance(nested, dict):
        return _as_identifier(nested.get(key))
    return ""


def _as_identifier(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _raise_event_error(payload: dict[str, Any]) -> None:
    error = payload.get("error")
    if error is None or error is False or error == "":
        if payload.get("type") != "error":
            if "detail" not in payload or "message" in payload:
                return
            error = payload.get("detail")
        else:
            error = payload.get("message") or payload

    if isinstance(error, dict):
        detail = error.get("message") or error.get("detail") or error.get("code")
    else:
        detail = error
    rendered = _safe_detail(detail)
    suffix = f": {rendered}" if rendered else ""
    raise BackendResponseError(f"ChatGPT backend stream reported an error{suffix}")


def _error_detail(body: str) -> str:
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return _safe_detail(body)
    if isinstance(value, dict):
        detail = value.get("detail") or value.get("message") or value.get("error")
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("detail") or detail.get("code")
        return _safe_detail(detail)
    return _safe_detail(value)


def _safe_detail(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            value = str(value)
    return " ".join(value.split())[:300]
