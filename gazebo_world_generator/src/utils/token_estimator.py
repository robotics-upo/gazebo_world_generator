"""Conservative token budgeting without a model download or executable tokenizer code."""

import math


def estimate_tokens(messages, chars_per_token: float = 3.0) -> int:
    """Upper-biased estimate for chat messages, including message framing."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return sum(
        math.ceil(len(str(message.get("content", "")).encode("utf-8")) /
                  chars_per_token) + 12
        for message in messages
    ) + 12
