"""Multi-modal input support — handle images, diagrams, and screenshots."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any


# Supported image MIME types for vision-capable models
SUPPORTED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
}

# Models known to support vision (image input)
_VISION_MODEL_PREFIXES = (
    "gpt-4o", "gpt-4-turbo", "gpt-4-vision",
    "claude-3", "claude-sonnet-4", "claude-opus-4", "claude-haiku-4",
    "gemini-",
    "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-opus-4-5", "claude-opus-4-6",
    "claude-haiku-4-5",
)


def supports_vision(model: str) -> bool:
    """Check if a model supports image/vision input."""
    model_lower = model.lower()
    # Bedrock/OpenRouter-hosted models
    stripped = model_lower
    for prefix in ("bedrock/converse/", "bedrock/", "openrouter/"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    # Strip region prefixes like "us.", "eu."
    if re.match(r"^[a-z]{2}\.", stripped):
        stripped = stripped[3:]
    # Strip provider prefixes like "anthropic."
    if stripped.startswith("anthropic."):
        stripped = stripped[len("anthropic."):]

    return any(stripped.startswith(p) for p in _VISION_MODEL_PREFIXES)


class ImageInput:
    """Represents an image input that can be sent to vision-capable models."""

    __slots__ = ("data", "media_type", "source")

    def __init__(
        self,
        data: bytes,
        media_type: str,
        source: str = "",
    ):
        self.data = data
        self.media_type = media_type
        self.source = source  # file path, URL, or "upload"

    @classmethod
    def from_file(cls, file_path: str) -> "ImageInput":
        """Load an image from a local file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        mime_type = mimetypes.guess_type(str(path))[0]
        if mime_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported image type: {mime_type}")

        data = path.read_bytes()
        return cls(data=data, media_type=mime_type, source=str(path))

    @classmethod
    def from_base64(cls, b64_data: str, media_type: str = "image/png") -> "ImageInput":
        """Create from a base64-encoded string."""
        # Strip data URL prefix if present
        if "," in b64_data:
            header, b64_data = b64_data.split(",", 1)
            # Extract MIME type from data URL
            mime_match = re.search(r"data:([^;]+)", header)
            if mime_match:
                media_type = mime_match.group(1)

        data = base64.b64decode(b64_data)
        return cls(data=data, media_type=media_type, source="base64")

    @classmethod
    def from_url(cls, url: str) -> "ImageInput":
        """Create a reference to an image URL (for models that support URL input)."""
        # We don't download — just store the URL reference
        # The LLM layer will handle URL-based images appropriately
        return cls(data=b"", media_type="image/url", source=url)

    def to_base64(self) -> str:
        """Encode the image data as base64."""
        return base64.b64encode(self.data).decode("utf-8")

    def to_content_block(self) -> dict[str, Any]:
        """Convert to an LLM-compatible content block.

        Returns the format expected by Claude/OpenAI for multi-modal messages:
        {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        """
        if self.media_type == "image/url":
            return {
                "type": "image_url",
                "image_url": {"url": self.source},
            }

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.to_base64(),
            },
        }


def build_multimodal_content(
    text: str,
    images: list[ImageInput] | None = None,
) -> str | list[dict[str, Any]]:
    """Build a message content field that may include text and images.

    If no images, returns plain text string (backward compatible).
    If images are present, returns a list of content blocks.
    """
    if not images:
        return text

    content_blocks: list[dict[str, Any]] = []

    # Add images first so the model sees them before the text prompt
    for img in images:
        content_blocks.append(img.to_content_block())

    # Add text block
    if text:
        content_blocks.append({"type": "text", "text": text})

    return content_blocks
