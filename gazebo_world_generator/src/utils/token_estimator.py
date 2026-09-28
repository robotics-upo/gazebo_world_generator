"""Conservative token budgeting without a model download or executable tokenizer code."""

import math

# Vision models tokenise an image to roughly this many tokens at our render sizes.
IMAGE_TOKENS = 1200


def message_text(content) -> str:
    """Text of a chat message whose content is a string or a list of parts."""
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
    return str(content or "")


def image_count(content) -> int:
    if isinstance(content, list):
        return sum(1 for part in content if isinstance(part, dict) and part.get("type") == "image_url")
    return 0


def estimate_tokens(messages, chars_per_token: float = 3.0) -> int:
    """Upper-biased estimate for chat messages, including message framing."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return sum(
        math.ceil(len(message_text(message.get("content", "")).encode("utf-8")) /
                  chars_per_token) + 12 + IMAGE_TOKENS * image_count(message.get("content"))
        for message in messages
    ) + 12
