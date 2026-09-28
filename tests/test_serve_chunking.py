"""Limits for long-form speech in the interactive demo."""

import pytest

from pocket_tts.serve_chunking import MAX_DEMO_TOKENS, split_demo_text


def count_words(text: str) -> int:
    return len(text.split())


def test_each_line_is_a_chunk_boundary():
    lines = [f"Line {number} is spoken." for number in range(1, 7)]

    assert split_demo_text("\n".join(lines), count_words) == lines


def test_nepali_danda_is_a_sentence_break():
    text = " ".join(f"यो वाक्य नम्बर {number} हो।" for number in range(12))

    chunks = split_demo_text(text, count_words)

    assert len(chunks) > 1
    assert all(count_words(chunk) <= MAX_DEMO_TOKENS for chunk in chunks)
    assert " ".join(chunks) == text


def test_unpunctuated_paragraph_is_split_at_word_boundaries():
    text = " ".join(f"word{number}" for number in range(80))

    chunks = split_demo_text(text, count_words)

    assert [count_words(chunk) for chunk in chunks] == [40, 40]
    assert " ".join(chunks) == text


def test_rejects_unsplittable_word():
    with pytest.raises(ValueError, match="One word exceeds"):
        split_demo_text("longword", lambda _: MAX_DEMO_TOKENS + 1)


def test_nepali_regression_paragraph_keeps_final_sentence_whole():
    text = (
        "कुकुर हाँशिरहेको थिएन, मुख सुकेको थियो। छोरालाई दया लाग्यो। "
        "उसले आफूसँग भएको खानेपानीबाट अलिकति पानी कचौरामा हालेर कुकुरलाई पिलायो।"
    )

    assert split_demo_text(text, count_words) == [text]


def test_rejects_excessive_chunks():
    with pytest.raises(ValueError, match="more than 80 audio chunks"):
        split_demo_text("\n".join("word" for _ in range(81)), count_words)
