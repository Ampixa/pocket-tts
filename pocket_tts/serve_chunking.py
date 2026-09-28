"""Bound long web-demo requests to short, independently generated utterances."""

import re
from collections.abc import Callable

# The model's own short-text limit is 50 tokens. A smaller server-only limit
# split ordinary Nepali sentences into one-word fragments and caused repeated
# end-of-utterance pauses. Leave room below the model limit while keeping
# complete short paragraphs together.
MAX_DEMO_TOKENS = 40
MAX_DEMO_CHUNKS = 80
MAX_DEMO_CHARACTERS = 12_000

_SENTENCE_BREAK = re.compile(r"(?<=[.!?।॥])\s+")


def split_demo_text(text: str, token_count: Callable[[str], int]) -> list[str]:
    """Keep each input line separate, never exceeding the token budget.

    Sentence boundaries are preferred inside a line. If a single sentence
    is too long, split at word boundaries; an unsplittable word is rejected
    explicitly instead of being passed to the model over budget.
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    if len(text) > MAX_DEMO_CHARACTERS:
        raise ValueError(f"Text exceeds {MAX_DEMO_CHARACTERS} characters; send it in parts")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Text cannot be empty")

    chunks: list[str] = []
    for line in lines:
        sentences = [part.strip() for part in _SENTENCE_BREAK.split(line) if part.strip()]
        current = ""

        def flush() -> None:
            nonlocal current
            if current:
                chunks.append(current)
                if len(chunks) > MAX_DEMO_CHUNKS:
                    raise ValueError(
                        f"Text needs more than {MAX_DEMO_CHUNKS} audio chunks; send it in parts"
                    )
                current = ""

        for sentence in sentences:
            if token_count(sentence) <= MAX_DEMO_TOKENS:
                candidate = f"{current} {sentence}".strip()
                if current and token_count(candidate) > MAX_DEMO_TOKENS:
                    flush()
                    candidate = sentence
                current = candidate
                continue

            flush()
            for word in sentence.split():
                if token_count(word) > MAX_DEMO_TOKENS:
                    raise ValueError(f"One word exceeds {MAX_DEMO_TOKENS} model tokens")
                candidate = f"{current} {word}".strip()
                if current and token_count(candidate) > MAX_DEMO_TOKENS:
                    flush()
                    candidate = word
                current = candidate
        flush()

    return chunks
