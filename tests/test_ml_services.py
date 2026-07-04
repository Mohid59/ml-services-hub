"""Transformer/CNN happy-path tests. Marked slow: first run downloads models.

Run all:      pytest
Skip slow:    pytest -m "not slow"
"""
import re

import pytest

slow = pytest.mark.slow


@slow
def test_ner_finds_entities(client):
    r = client.post("/ner/", data={"text": "Barack Obama joined Google in "
                                           "London on Monday."})
    assert b"ent-PER" in r.data and b"ent-ORG" in r.data
    assert b"flash-error" not in r.data


@slow
@pytest.mark.parametrize("text,label", [
    ("This is absolutely wonderful!", b"POSITIVE"),
    ("This is terrible and awful.", b"NEGATIVE"),
])
def test_sentiment_text_fallback(client, text, label):
    r = client.post("/sentiment/", data={"text": text})
    assert label in r.data
    assert b"flash-error" not in r.data


@slow
def test_translation_produces_urdu(client):
    r = client.post("/translation/", data={"text": "Good morning"})
    html = r.data.decode()
    m = re.search(r'class="urdu" dir="rtl">(.*?)</p>', html, re.S)
    assert m, "urdu block missing"
    urdu_chars = sum(1 for ch in m.group(1) if "؀" <= ch <= "ۿ")
    assert urdu_chars >= 3


@slow
def test_textgen_output_is_tidy(client):
    r = client.post("/textgen/", data={"prompt": "The history of Rome",
                                       "max_new_tokens": "30",
                                       "temperature": "0.8"})
    html = r.data.decode()
    m = re.search(r'class="gen-out">(.*?)</p>', html, re.S)
    assert m, "generation output missing"
    body = re.sub(r"<[^>]+>", "", m.group(1))
    assert "\n\n" not in body          # no blank-line runs
    assert body == body.lstrip()       # no leading whitespace gap


@slow
def test_qa_returns_answer_and_audio(client):
    r = client.post("/qa/", data={"question": "What is Flask?", "context": ""})
    assert b"data:audio/mp3;base64," in r.data
    assert b"flash-error" not in r.data


@slow
def test_gender_predicts_on_jpeg(client, jpeg_bytes):
    r = client.post("/gender/", data={"imagefile": (jpeg_bytes, "f.jpg")},
                    content_type="multipart/form-data")
    assert b"predicted" in r.data
    assert b"flash-error" not in r.data
