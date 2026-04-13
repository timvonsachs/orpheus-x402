"""HTTP-Routen fuer Orpheus Acoustic Sense."""

from __future__ import annotations

import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from api.schemas import HealthResponse, SenseMode, SenseResponse
from config import settings
from engine.alerts import AlertDetector
from engine.biomarkers import BiomarkerEngine
from engine.combiner import AcousticSenseCombiner
from engine.humanness import HumannessEngine
from engine.paralinguistic import ParalinguisticInterpreter
from engine.trends import TrendAnalyzer

router = APIRouter()


@lru_cache(maxsize=1)
def get_combiner() -> AcousticSenseCombiner:
    humanness_engine = HumannessEngine(
        checkpoint_path=settings.model_checkpoint,
        device=settings.device,
    )
    biomarker_engine = BiomarkerEngine(orpheus_api_url=settings.orpheus_api_url)
    interpreter = ParalinguisticInterpreter()
    trend_analyzer = TrendAnalyzer()
    alert_detector = AlertDetector()
    return AcousticSenseCombiner(
        humanness_engine=humanness_engine,
        biomarker_engine=biomarker_engine,
        interpreter=interpreter,
        trend_analyzer=trend_analyzer,
        alert_detector=alert_detector,
        version=settings.version,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=settings.version)


@router.post("/v1/sense", response_model=SenseResponse)
async def sense(
    file: UploadFile = File(...),
    mode: SenseMode = Query("full"),
    segments: bool = Query(True),
    segment_duration: float = Query(5.0, ge=1.0, le=30.0),
    diarize: bool = Query(False),
    transcribe: bool = Query(False),
    environment: bool = Query(True),
) -> SenseResponse:
    start_time = time.perf_counter()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Dateiname fehlt im Upload.")

    suffix = Path(file.filename).suffix or ".wav"
    tmp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Upload-Datei ist leer.")
            tmp.write(content)
            tmp_path = tmp.name

        result = await get_combiner().analyze(
            audio_path=tmp_path,
            mode=mode,
            include_segments=segments,
            segment_duration=segment_duration,
            diarize=diarize,
            transcribe=transcribe,
            detect_environment=environment,
        )
        result["processing_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
        return SenseResponse.model_validate(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analyse fehlgeschlagen: {exc}") from exc
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
