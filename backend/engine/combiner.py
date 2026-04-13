"""Kombiniert beide Ohren (Humanness + Biomarker) in ein Antwortobjekt."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import os

import torch
import numpy as np
import torchaudio
import wave

from engine.diarization import SpeakerDiarizer
from engine.environment import EnvironmentDetector
from engine.keyword_emotion import KeywordEmotionAnalyzer
from engine.preprocessing import AudioPreprocessor
from engine.transcription import Transcriber


class AcousticSenseCombiner:
    def __init__(self, humanness_engine, biomarker_engine, interpreter, trend_analyzer, alert_detector, version: str):
        self.humanness = humanness_engine
        self.biomarkers = biomarker_engine
        self.interpreter = interpreter
        self.trends = trend_analyzer
        self.alerts = alert_detector
        self.version = version
        self.preprocessor = AudioPreprocessor()
        self.diarizer = SpeakerDiarizer()
        self.environment = EnvironmentDetector()
        self.keyword_emotion = KeywordEmotionAnalyzer()
        self.transcriber = Transcriber(model_size="base")

    async def analyze(
        self,
        audio_path: str,
        mode: str = "full",
        include_segments: bool = True,
        segment_duration: float = 5.0,
        speaker_id: str = "default",
        diarize: bool = False,
        transcribe: bool = False,
        detect_environment: bool = True,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"orpheus_version": self.version}
        processed_path = audio_path
        created_temp = False

        try:
            processed_path = self.preprocessor.process_temp(audio_path)
            created_temp = processed_path != audio_path

            waveform = None
            sr = None
            if mode in {"full", "agent"} or include_segments:
                waveform, sr = self._load_waveform(processed_path)
                duration = waveform.shape[-1] / float(sr)
                result["audio_duration_seconds"] = round(duration, 3)

            humanness_segments: list[dict[str, Any]] = []
            if detect_environment:
                try:
                    result["environment"] = self.environment.detect(processed_path)
                except Exception:
                    # Environment detection should never break batch analysis.
                    pass

            if mode in {"full", "agent"} and waveform is not None and sr is not None:
                humanness = self.humanness.score_audio(waveform, sr)
                result["humanness"] = humanness
                if include_segments:
                    humanness_segments = self.humanness.score_segments(
                        waveform=waveform,
                        sr=sr,
                        segment_duration=segment_duration,
                    )

            if mode in {"full", "human"}:
                bio_response = await self.biomarkers.extract(
                    audio_path=processed_path,
                    include_segments=include_segments,
                    segment_duration=segment_duration,
                )

                biomarkers = bio_response.get("biomarkers", {})
                paralinguistic_summary = self.interpreter.interpret(biomarkers)
                result["paralinguistic"] = {
                    "summary": paralinguistic_summary,
                    "biomarkers": biomarkers,
                    "voice_profile": bio_response.get("voice_dna", bio_response.get("voice_profile", {})),
                }

                if include_segments:
                    incoming_segments = bio_response.get("segments", [])
                    processed_segments = []
                    for idx, seg in enumerate(incoming_segments):
                        seg_bio = seg.get("features", seg.get("biomarkers", {}))
                        seg_para = self.interpreter.interpret(seg_bio)
                        item = {
                            "start": float(seg.get("start_seconds", seg.get("start", 0))),
                            "end": float(seg.get("end_seconds", seg.get("end", 0))),
                            "biomarkers": seg_bio,
                            "paralinguistic_summary": seg_para,
                            "speaker": "unknown",
                        }
                        if idx < len(humanness_segments):
                            item["humanness_score"] = humanness_segments[idx]["humanness_score"]
                        processed_segments.append(item)

                    if mode == "full" and not processed_segments and humanness_segments:
                        processed_segments = humanness_segments

                    result["segments"] = processed_segments

                    if len(processed_segments) >= 2:
                        result["trends"] = self.trends.analyze(processed_segments)
                        speaker_key = speaker_id or Path(audio_path).stem or "default"
                        result["alerts"] = self.alerts.detect(processed_segments, speaker_id=speaker_key)
                        self._attach_alerts_to_segments(processed_segments, result.get("alerts", []))

                    if diarize and processed_segments:
                        diarized_segments = self.diarizer.diarize(processed_path)
                        result["diarization"] = diarized_segments
                        role_segments = self._attach_roles(processed_segments, diarized_segments)

                        customer_segments = [s for s in role_segments if s.get("role") == "customer"]
                        agent_segments = [s for s in role_segments if s.get("role") == "agent"]

                        customer_alerts: list[dict[str, Any]] = []
                        if len(customer_segments) >= 2:
                            customer_alerts = self.alerts.detect(
                                customer_segments,
                                speaker_id=f"{speaker_key}_customer",
                            )

                        result["customer"] = {
                            "segments": customer_segments,
                            "avg_stress": self._avg_metric(customer_segments, "stress_level"),
                            "avg_engagement": self._avg_metric(customer_segments, "engagement"),
                            "alerts": customer_alerts,
                        }
                        result["agent"] = {
                            "segments": agent_segments,
                            "avg_stress": self._avg_metric(agent_segments, "stress_level"),
                            "avg_engagement": self._avg_metric(agent_segments, "engagement"),
                        }
                        result["turn_taking"] = self._turn_taking_stats(diarized_segments)

                    if transcribe:
                        transcription = self.transcriber.transcribe(processed_path)
                        result["transcription"] = transcription
                        self._attach_text_to_segments(processed_segments, transcription)
                        result["keyword_emotion"] = self.keyword_emotion.analyze(processed_segments)

            elif include_segments and humanness_segments:
                result["segments"] = humanness_segments

            return result
        finally:
            if created_temp and processed_path and os.path.exists(processed_path):
                try:
                    os.unlink(processed_path)
                except OSError:
                    pass

    @staticmethod
    def _attach_text_to_segments(segments: list[dict[str, Any]], transcription: list[dict[str, Any]]) -> None:
        if not segments:
            return
        for seg in segments:
            s0 = float(seg.get("start", 0.0) or 0.0)
            s1 = float(seg.get("end", 0.0) or 0.0)
            texts: list[str] = []
            for ts in transcription:
                t0 = float(ts.get("start", 0.0) or 0.0)
                t1 = float(ts.get("end", 0.0) or 0.0)
                overlap = max(0.0, min(s1, t1) - max(s0, t0))
                if overlap > 0:
                    text = str(ts.get("text", "") or "").strip()
                    if text:
                        texts.append(text)
            seg["text"] = " ".join(texts).strip()

    @staticmethod
    def _attach_alerts_to_segments(segments: list[dict[str, Any]], alerts: list[dict[str, Any]]) -> None:
        for seg in segments:
            seg["alerts"] = []
        if not alerts:
            return
        for alert in alerts:
            ts = float(alert.get("timestamp", 0.0) or 0.0)
            atype = str(alert.get("type", "alert"))
            for seg in segments:
                s0 = float(seg.get("start", 0.0) or 0.0)
                s1 = float(seg.get("end", 0.0) or 0.0)
                if s0 <= ts <= s1:
                    seg["alerts"].append(atype)
                    break

    @staticmethod
    def _load_waveform(audio_path: str) -> tuple[torch.Tensor, int]:
        try:
            return torchaudio.load(audio_path)
        except Exception:
            try:
                import soundfile as sf

                data, sr = sf.read(audio_path, always_2d=True, dtype="float32")
                data = data.T  # (time, channels) -> (channels, time)
                return torch.from_numpy(data), int(sr)
            except Exception:
                pass

            with wave.open(audio_path, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

            if sample_width == 1:
                arr = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                arr = (arr - 128.0) / 128.0
            elif sample_width == 2:
                arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                arr = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported WAV sample width: {sample_width}")

            if channels > 1:
                arr = arr.reshape(-1, channels).T
            else:
                arr = arr.reshape(1, -1)
            return torch.from_numpy(arr), sr

    @staticmethod
    def _avg_metric(segments: list[dict[str, Any]], key: str) -> float:
        values = []
        for seg in segments:
            para = seg.get("paralinguistic_summary", {}) or {}
            value = para.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            return 0.0
        return round(sum(values) / len(values), 3)

    @staticmethod
    def _attach_roles(
        analysis_segments: list[dict[str, Any]],
        diarized_segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        speech = [s for s in diarized_segments if s.get("role") in {"agent", "customer"}]
        if not speech:
            return analysis_segments

        enriched: list[dict[str, Any]] = []
        for seg in analysis_segments:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", 0.0) or 0.0)
            midpoint = (start + end) / 2
            role = "unknown"

            for d in speech:
                ds = float(d.get("start", 0.0) or 0.0)
                de = float(d.get("end", 0.0) or 0.0)
                if ds <= midpoint <= de:
                    role = str(d.get("role", "unknown"))
                    break

            if role == "unknown":
                overlaps: list[tuple[float, str]] = []
                for d in speech:
                    ds = float(d.get("start", 0.0) or 0.0)
                    de = float(d.get("end", 0.0) or 0.0)
                    overlap = max(0.0, min(end, de) - max(start, ds))
                    if overlap > 0:
                        overlaps.append((overlap, str(d.get("role", "unknown"))))
                if overlaps:
                    overlaps.sort(key=lambda x: x[0], reverse=True)
                    role = overlaps[0][1]

            item = dict(seg)
            item["role"] = role
            item["speaker"] = role
            enriched.append(item)
        return enriched

    @staticmethod
    def _turn_taking_stats(diarized_segments: list[dict[str, Any]]) -> dict[str, Any]:
        speech = [s for s in diarized_segments if s.get("role") in {"agent", "customer"}]
        if not speech:
            return {
                "total_turns": 0,
                "customer_talk_ratio": 0.0,
                "avg_customer_turn_length": 0.0,
                "interruptions": 0,
            }

        total_speech_time = 0.0
        customer_time = 0.0
        customer_turn_lengths: list[float] = []
        interruptions = 0

        prev_role: Optional[str] = None
        prev_end: Optional[float] = None
        for seg in speech:
            role = str(seg.get("role"))
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", 0.0) or 0.0)
            dur = max(0.0, end - start)
            total_speech_time += dur
            if role == "customer":
                customer_time += dur
                customer_turn_lengths.append(dur)

            if prev_role is not None and role != prev_role and prev_end is not None:
                gap = max(0.0, start - prev_end)
                if gap < 0.2:
                    interruptions += 1
            prev_role = role
            prev_end = end

        avg_customer_turn = (sum(customer_turn_lengths) / len(customer_turn_lengths)) if customer_turn_lengths else 0.0
        return {
            "total_turns": len(speech),
            "customer_talk_ratio": round(customer_time / total_speech_time, 3) if total_speech_time > 0 else 0.0,
            "avg_customer_turn_length": round(avg_customer_turn, 3),
            "interruptions": interruptions,
        }
