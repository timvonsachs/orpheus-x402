"""Realtime streaming endpoint for chunk-level emotional and humanness feedback."""

from __future__ import annotations

from functools import lru_cache
import io
import math
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import torch

from config import settings
from engine.baselines import SpeakerBaseline
from engine.diarization import StreamDiarizer
from engine.environment import EnvironmentDetector
from engine.humanness import HumannessEngine
from engine.humanness_adapter import HumannessAdapter
from engine.paralinguistic import ParalinguisticInterpreter
from engine.strategy import ConversationStrategy
from engine.transcription import Transcriber
from engine.tts_adapter import TTSAdapter

router = APIRouter()

TARGET_SR = 16000
CHUNK_SECONDS = 0.5
ENV_REFRESH_SECONDS = 5.0
DEFAULT_BASELINE_DIR = Path(__file__).resolve().parent.parent / "data" / "baselines"


@lru_cache(maxsize=1)
def get_stream_humanness_engine() -> Optional[HumannessEngine]:
    try:
        checkpoint = Path(settings.model_checkpoint)
        if not checkpoint.exists():
            root_ckpt = Path(__file__).resolve().parent.parent / settings.model_checkpoint
            checkpoint = root_ckpt if root_ckpt.exists() else checkpoint
        if not checkpoint.exists():
            return None
        return HumannessEngine(checkpoint_path=str(checkpoint), device=settings.device, max_duration=0.5)
    except Exception:
        return None


class StreamingAnalyzer:
    def __init__(
        self,
        speaker_id: str,
        transcribe: bool = False,
        baseline_dir: Optional[Path] = None,
        alpha: float = 0.3,
        warmup_segments: int = 5,
    ):
        self.speaker_id = speaker_id
        self.baseline_dir = baseline_dir or DEFAULT_BASELINE_DIR
        self.interpreter = ParalinguisticInterpreter()
        self.tts_adapter = TTSAdapter()
        self.strategy = ConversationStrategy()
        self.humanness_adapter = HumannessAdapter()
        self.stream_diarizer = StreamDiarizer()
        self.transcribe_enabled = transcribe
        self.transcriber = Transcriber(model_size="base") if transcribe else None
        self.environment_detector = EnvironmentDetector()
        self.baseline = SpeakerBaseline.load_from_path(
            speaker_id=speaker_id,
            base_dir=self.baseline_dir,
            alpha=alpha,
            warmup_segments=warmup_segments,
        )
        self.stress_history: list[float] = []
        self.humanness_scores: list[float] = []
        self.current_time = 0.0
        self.last_text = ""
        self.env_cache: Optional[dict] = None
        self.next_env_update_time = 0.0
        self.env_buffer = np.asarray([], dtype=np.float32)

    def process_chunk(self, payload: bytes) -> dict:
        samples, sample_rate = self._decode_audio_chunk(payload)
        samples = self._to_mono(samples)
        if sample_rate != TARGET_SR:
            samples = self._resample(samples, sample_rate, TARGET_SR)
            sample_rate = TARGET_SR

        chunk_duration = len(samples) / float(sample_rate) if sample_rate > 0 else CHUNK_SECONDS
        self.current_time += chunk_duration
        timestamp = round(self.current_time, 3)
        self._append_env_buffer(samples)

        biomarkers = self._extract_biomarkers(samples, sample_rate)
        speaker = self.stream_diarizer.identify(samples)
        self._maybe_update_environment()

        response: dict = {"timestamp": timestamp, "speaker": speaker, "alert": "none"}
        if self.env_cache is not None:
            response["environment"] = {
                "detected": self.env_cache.get("environment", "unknown"),
                "confidence": self.env_cache.get("confidence", 0.0),
                "noise_level": self.env_cache.get("noise_level", "moderate"),
                "strategy_impact": self.env_cache.get("strategy_impact", {}),
            }

        if speaker in {"customer", "both"}:
            customer_block = self._customer_state_block(biomarkers)
            response.update(customer_block)

        if speaker in {"agent", "both"}:
            humanness_score = self._score_humanness(samples, biomarkers)
            self.humanness_scores.append(humanness_score)
            agent_block = self.humanness_adapter.analyze(samples, humanness_score)
            agent_block["agent_state"]["trend"] = self.humanness_adapter.trend(self.humanness_scores)
            response.update(agent_block)

        if self.transcribe_enabled:
            response["text"] = self._transcribe_window(samples)

        self.baseline.save_to_path(self.baseline_dir)
        return response

    def _customer_state_block(self, biomarkers: dict[str, float]) -> dict:
        para = self.interpreter.interpret(biomarkers)
        stress = float(para.get("stress_level", 0.0) or 0.0)
        engagement = float(para.get("engagement", 0.0) or 0.0)
        arousal = float(para.get("arousal", 0.0) or 0.0)

        z_stress = float(self.baseline.z_score("stress", stress))
        trend = self._stress_trend(stress)

        recommendation = self.strategy.recommend(
            stress=stress,
            engagement=engagement,
            trend=trend,
            z_score=z_stress,
            environment=(self.env_cache or {}).get("environment"),
            noise_level=(self.env_cache or {}).get("noise_level"),
        )
        tts_params = self.tts_adapter.recommend(
            stress=stress,
            engagement=engagement,
            arousal=arousal,
            z_score_stress=z_stress,
        )
        alert = self._alert_name(stress=stress, engagement=engagement, trend=trend, z_stress=z_stress)

        self._update_baseline(biomarkers=biomarkers, stress=stress, engagement=engagement)
        return {
            "customer_state": {
                "stress": round(stress, 3),
                "engagement": round(engagement, 3),
                "arousal": round(arousal, 3),
                "trend": trend,
                "z_score_stress": round(z_stress, 3),
            },
            "alert": alert,
            "recommendation": recommendation,
            "tts_params": tts_params,
        }

    def _append_env_buffer(self, samples: np.ndarray) -> None:
        self.env_buffer = np.concatenate([self.env_buffer, samples.astype(np.float32)])
        max_len = int(TARGET_SR * max(ENV_REFRESH_SECONDS, 6.0))
        if len(self.env_buffer) > max_len:
            self.env_buffer = self.env_buffer[-max_len:]

    def _maybe_update_environment(self) -> None:
        if self.current_time < self.next_env_update_time and self.env_cache is not None:
            return
        if len(self.env_buffer) < int(TARGET_SR * 1.0):
            return
        try:
            self.env_cache = self.environment_detector.detect_chunk(self.env_buffer, sr=TARGET_SR)
            self.next_env_update_time = self.current_time + ENV_REFRESH_SECONDS
        except Exception:
            # Environment block is optional and must not break existing streaming behavior.
            pass

    def _transcribe_window(self, samples: np.ndarray) -> str:
        if self.transcriber is None:
            return self.last_text
        try:
            out = self.transcriber.transcribe_chunk(samples, sr=TARGET_SR)
            if out.get("ready") and out.get("text"):
                self.last_text = str(out.get("text", "")).strip()
        except Exception:
            pass
        return self.last_text

    def _score_humanness(self, samples: np.ndarray, biomarkers: dict[str, float]) -> float:
        engine = get_stream_humanness_engine()
        if engine is not None:
            try:
                waveform = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
                out = engine.score_audio(waveform, TARGET_SR)
                return float(out.get("score", 0.0) or 0.0)
            except Exception:
                pass
        return self._dummy_humanness_score(biomarkers)

    @staticmethod
    def _dummy_humanness_score(biomarkers: dict[str, float]) -> float:
        score = 85.0
        jitter = float(biomarkers.get("jitter", 0.0) or 0.0)
        f0_var = float(biomarkers.get("f0_var", 0.0) or 0.0)
        speech_rate = float(biomarkers.get("speech_rate", 0.0) or 0.0)
        pause_rate = float(biomarkers.get("pause_rate", 0.0) or 0.0)
        loudness = float(biomarkers.get("loudness_rms", 0.0) or 0.0)

        if jitter < 0.01:
            score -= 22.0
        if f0_var < 10.0:
            score -= 24.0
        if 3.5 <= speech_rate <= 5.5:
            score -= 8.0
        if 18.0 <= pause_rate <= 32.0:
            score -= 8.0
        if loudness < 0.015:
            score -= 8.0
        return max(5.0, min(98.0, score))

    def _update_baseline(self, biomarkers: dict, stress: float, engagement: float) -> None:
        self.baseline.update("stress", stress)
        self.baseline.update("engagement", engagement)
        self.baseline.update("f0_mean", float(biomarkers.get("f0_mean", 0.0) or 0.0))
        self.baseline.update("jitter", float(biomarkers.get("jitter", 0.0) or 0.0))
        self.baseline.update("speech_rate", float(biomarkers.get("speech_rate", 0.0) or 0.0))
        self.baseline.update("pause_rate", float(biomarkers.get("pause_rate", 0.0) or 0.0))

    def _stress_trend(self, stress: float) -> str:
        self.stress_history.append(stress)
        window = self.stress_history[-4:]
        if len(window) < 2:
            return "stable"
        delta = window[-1] - window[0]
        if delta > 0.06:
            return "escalating"
        if delta < -0.06:
            return "deescalating"
        return "stable"

    @staticmethod
    def _alert_name(stress: float, engagement: float, trend: str, z_stress: float) -> str:
        if stress > 0.7 and trend == "escalating":
            return "frustration_rising"
        if z_stress > 2.0:
            return "stress_anomaly"
        if engagement < 0.3:
            return "engagement_drop"
        return "none"

    @staticmethod
    def _decode_audio_chunk(payload: bytes) -> tuple[np.ndarray, int]:
        try:
            import soundfile as sf

            data, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
            return np.asarray(data, dtype=np.float32), int(sr)
        except Exception:
            pcm = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
            if pcm.size == 0:
                raise ValueError("Leerer Audio-Chunk empfangen.")
            return pcm.reshape(-1, 1), TARGET_SR

    @staticmethod
    def _to_mono(samples: np.ndarray) -> np.ndarray:
        if samples.ndim == 1:
            return samples.astype(np.float32)
        if samples.shape[1] == 1:
            return samples[:, 0].astype(np.float32)
        return samples.mean(axis=1).astype(np.float32)

    @staticmethod
    def _resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
            return samples.astype(np.float32)
        duration = len(samples) / float(src_sr)
        dst_len = max(1, int(round(duration * dst_sr)))
        src_x = np.linspace(0.0, duration, num=len(samples), endpoint=False)
        dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
        return np.interp(dst_x, src_x, samples).astype(np.float32)

    def _extract_biomarkers(self, samples: np.ndarray, sample_rate: int) -> dict[str, float]:
        frame_length = max(1, int(0.025 * sample_rate))
        hop = max(1, int(0.01 * sample_rate))
        if len(samples) < frame_length:
            samples = np.pad(samples, (0, frame_length - len(samples)))

        frames = []
        for start in range(0, len(samples) - frame_length + 1, hop):
            frames.append(samples[start : start + frame_length])
        framed = np.asarray(frames, dtype=np.float32) if frames else np.asarray([samples[:frame_length]], dtype=np.float32)
        rms = np.sqrt(np.mean(framed**2, axis=1) + 1e-9)

        threshold = max(0.01, float(np.median(rms)) * 1.2)
        speech_mask = rms >= threshold
        speech_ratio = float(np.mean(speech_mask))
        pause_ratio = 1.0 - speech_ratio
        speech_onsets = int(np.sum((~speech_mask[:-1]) & speech_mask[1:])) if len(speech_mask) > 1 else int(speech_mask[0])
        chunk_seconds = len(samples) / float(sample_rate)
        onsets_per_sec = speech_onsets / max(chunk_seconds, 1e-6)

        speech_rate = max(0.5, min(8.0, 2.0 + speech_ratio * 4.5 + onsets_per_sec * 0.2))
        pause_rate = max(0.0, min(50.0, pause_ratio * 50.0))
        pause_dur = pause_ratio * chunk_seconds

        f0_track = self._estimate_f0_track(samples, sample_rate)
        voiced = f0_track[~np.isnan(f0_track)] if f0_track.size else np.asarray([], dtype=np.float32)
        f0_mean = float(np.mean(voiced)) if voiced.size else 0.0
        f0_var = float(np.var(voiced)) if voiced.size else 0.0
        if voiced.size >= 2:
            f0_range_st = float(12.0 * math.log2(max(np.max(voiced), 1e-6) / max(np.min(voiced), 1e-6)))
            jitter = float(np.mean(np.abs(np.diff(voiced)) / np.maximum(voiced[:-1], 1e-6)))
        else:
            f0_range_st = 0.0
            jitter = 0.0

        speech_rms = rms[speech_mask] if np.any(speech_mask) else rms
        if speech_rms.size >= 2:
            shimmer = float(np.mean(np.abs(np.diff(speech_rms))) / max(np.mean(speech_rms), 1e-6))
        else:
            shimmer = 0.0

        smoothed = np.convolve(samples, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        harmonic_power = float(np.mean(smoothed**2))
        noise_power = float(np.mean((samples - smoothed) ** 2)) + 1e-8
        hnr = max(0.0, min(20.0, 10.0 * math.log10(harmonic_power / noise_power)))

        return {
            "f0_mean": float(round(f0_mean, 3)),
            "f0_var": float(round(f0_var, 3)),
            "f0_range_st": float(round(max(f0_range_st, 0.0), 3)),
            "jitter": float(round(max(jitter, 0.0), 4)),
            "shimmer": float(round(max(shimmer, 0.0), 4)),
            "hnr": float(round(hnr, 3)),
            "speech_rate": float(round(speech_rate, 3)),
            "pause_rate": float(round(pause_rate, 3)),
            "pause_dur": float(round(pause_dur, 3)),
            "loudness_rms": float(round(float(np.sqrt(np.mean(samples**2) + 1e-9)), 6)),
        }

    @staticmethod
    def _estimate_f0_track(samples: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import librosa

            f0, _, _ = librosa.pyin(
                samples.astype(np.float64),
                sr=sample_rate,
                fmin=65,
                fmax=400,
                frame_length=1024,
                hop_length=160,
            )
            if f0 is None:
                return np.asarray([], dtype=np.float32)
            return np.asarray(f0, dtype=np.float32)
        except Exception:
            return np.asarray([], dtype=np.float32)


@router.websocket("/v1/stream")
async def stream(websocket: WebSocket) -> None:
    speaker_id = (
        websocket.headers.get("speaker_id")
        or websocket.headers.get("x-speaker-id")
        or websocket.query_params.get("speaker_id")
        or "default"
    )
    transcribe = str(websocket.query_params.get("transcribe", "false")).lower() in {"1", "true", "yes", "on"}
    analyzer = StreamingAnalyzer(speaker_id=speaker_id, transcribe=transcribe)
    await websocket.accept()

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            payload = msg.get("bytes")
            if payload is None:
                text = msg.get("text")
                if text in {"close", "stop"}:
                    await websocket.close()
                    return
                await websocket.send_json({"error": "Bitte Audio als Binary-Chunk senden."})
                continue

            try:
                response = analyzer.process_chunk(payload)
                await websocket.send_json(response)
            except Exception as exc:
                await websocket.send_json({"error": f"Chunk-Analyse fehlgeschlagen: {exc}"})
    except WebSocketDisconnect:
        return
