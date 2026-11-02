"""Safe local-image intake and optional vision-model analysis."""
from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aether.config import load_settings
from aether.ollama import ModelUnavailable
from aether.providers import ProviderChain, build_registry
from aether.resources import ResourceGuard, ResourcePolicy


MAX_IMAGE_BYTES = 8 * 1024 * 1024


def detect_image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported image format; use PNG, JPEG, GIF or WebP (SVG is rejected)")


def load_image_payload(path: str | Path, *, max_bytes: int = MAX_IMAGE_BYTES) -> dict[str, str]:
    image_path = Path(path).expanduser().resolve(strict=True)
    if not image_path.is_file():
        raise ValueError("Image path must point to a regular file")
    size = image_path.stat().st_size
    if size < 1:
        raise ValueError("Image file is empty")
    if size > max_bytes:
        raise ValueError(f"Image exceeds the {max_bytes}-byte safety limit")
    data = image_path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f"Image exceeds the {max_bytes}-byte safety limit")
    mime_type = detect_image_mime(data)
    return {
        "mime_type": mime_type,
        "data": base64.b64encode(data).decode("ascii"),
        "filename": image_path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def analyze_image(provider: Any, path: str | Path, prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A non-empty image-analysis prompt is required")
    image = load_image_payload(path)
    method = getattr(provider, "chat_with_images", None)
    if not callable(method):
        raise ModelUnavailable("Configured provider chain does not support image analysis")
    return method(prompt.strip(), [image])


def _is_loopback_url(value: str) -> bool:
    hostname = (urlparse(value).hostname or "").casefold()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a local image with a configured vision model.")
    parser.add_argument("--image", required=True, help="Local PNG/JPEG/GIF/WebP image path")
    parser.add_argument("--prompt", default="Describe only what is visibly supported by the image.")
    parser.add_argument(
        "--allow-remote", action="store_true",
        help="Explicitly permit sending the image to a non-local OpenAI-compatible endpoint",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    if not settings.models.enabled:
        raise ModelUnavailable("Enable [models].enabled before image analysis")
    remote_endpoints = []
    if settings.models.provider == "openai_compatible" and not _is_loopback_url(settings.models.base_url):
        remote_endpoints.append(settings.models.base_url)
    for fallback in settings.models.fallbacks:
        if fallback.provider == "openai_compatible" and not _is_loopback_url(fallback.base_url):
            remote_endpoints.append(fallback.base_url)
    if remote_endpoints and not args.allow_remote:
        raise PermissionError(
            "A configured model endpoint is remote. Re-run with --allow-remote only if you consent "
            "to sending this image to that endpoint."
        )

    registry = build_registry(
        settings.models.base_url, settings.models.model,
        settings.models.timeout_seconds, settings.models.provider,
        settings.models.api_key_env, settings.models.fallbacks,
    )
    providers = [(settings.models.provider, registry.get(settings.models.provider))]
    providers.extend(
        (f"fallback_{index}", registry.get(f"fallback_{index}"))
        for index in range(1, len(settings.models.fallbacks) + 1)
    )
    chain = ProviderChain(providers)
    guard = ResourceGuard(ResourcePolicy(**settings.resources.__dict__))
    guard.check(
        "image analysis",
        gpu_required=(settings.models.provider == "ollama" and not settings.models.fallbacks),
    )
    result = analyze_image(chain, args.image, args.prompt)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
