"""Proxy-Engine fuer Biomarker-Extraktion via bestehender Orpheus API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class BiomarkerEngine:
    def __init__(self, orpheus_api_url: str, timeout_seconds: float = 120.0):
        self.api_url = orpheus_api_url
        self.timeout_seconds = timeout_seconds

    async def extract(
        self,
        audio_path: str,
        include_segments: bool = True,
        segment_duration: float = 5.0,
    ) -> dict[str, Any]:
        path = Path(audio_path)
        mime = self._guess_mime_type(path.suffix.lower())
        params = {"segments": str(include_segments).lower(), "segment_duration": segment_duration}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            with path.open("rb") as fp:
                response = await client.post(
                    self.api_url,
                    params=params,
                    files={"file": (path.name, fp, mime)},
                )
            response.raise_for_status()
            payload = response.json()

        if isinstance(payload, dict):
            return payload
        return {}

    @staticmethod
    def _guess_mime_type(suffix: str) -> str:
        return {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
        }.get(suffix, "application/octet-stream")
