"""Pydantic-Schemas fuer API-Antworten."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    ears: list[str] = Field(default_factory=lambda: ["humanness", "paralinguistic", "environment"])


class SenseResponse(BaseModel):
    orpheus_version: str
    audio_duration_seconds: Optional[float] = None
    processing_time_ms: float

    humanness: Optional[dict[str, Any]] = None
    paralinguistic: Optional[dict[str, Any]] = None
    segments: Optional[list[dict[str, Any]]] = None
    trends: Optional[dict[str, Any]] = None
    alerts: Optional[list[dict[str, Any]]] = None
    environment: Optional[dict[str, Any]] = None
    transcription: Optional[list[dict[str, Any]]] = None
    keyword_emotion: Optional[dict[str, Any]] = None
    diarization: Optional[list[dict[str, Any]]] = None
    customer: Optional[dict[str, Any]] = None
    agent: Optional[dict[str, Any]] = None
    turn_taking: Optional[dict[str, Any]] = None


SenseMode = Literal["full", "human", "agent"]
