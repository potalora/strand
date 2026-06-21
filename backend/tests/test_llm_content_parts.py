from __future__ import annotations

from app.services.ai.llm.types import (
    DocumentPart,
    ImagePart,
    LLMMessage,
    TextPart,
    as_parts,
)


def test_str_content_normalizes_to_textpart():
    assert as_parts("hello") == [TextPart("hello")]


def test_list_content_passthrough():
    img = ImagePart(b"\x89PNG", "image/png")
    parts = as_parts([TextPart("describe"), img])
    assert parts[1] is img


def test_message_accepts_parts():
    m = LLMMessage("user", [DocumentPart(b"%PDF", "application/pdf")])
    assert isinstance(m.content, list)


def test_message_accepts_str():
    m = LLMMessage("user", "plain text still works")
    assert as_parts(m.content) == [TextPart("plain text still works")]
