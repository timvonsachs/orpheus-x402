"""Lightweight two-speaker diarization for call audio."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchaudio


class SpeakerDiarizer:
    """Einfache energie-basierte Speaker-Segmentierung fuer Zwei-Personen-Calls."""

    def __init__(self, energy_threshold: float = 0.02, min_segment_duration: float = 0.5):
        self.energy_threshold = energy_threshold
        self.min_segment_duration = min_segment_duration
        self.sample_rate = 16000
        self.frame_duration = 0.05  # 50ms

    def diarize(self, audio_path: str) -> list[dict[str, Any]]:
        waveform, sr = self._load_audio(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            sr = self.sample_rate

        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        data = waveform.detach().cpu().numpy().astype(np.float32)
        frame_samples = max(1, int(sr * self.frame_duration))
        n_frames = math.ceil(len(data) / frame_samples)
        padded_len = n_frames * frame_samples
        if padded_len > len(data):
            data = np.pad(data, (0, padded_len - len(data)))

        frames = data.reshape(n_frames, frame_samples)
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        speech_mask = rms >= self.energy_threshold
        segments = self._segments_from_mask(speech_mask, rms, sr, frame_samples, data)
        return self.label_speakers(segments, method="f0_based")

    def label_speakers(self, segments: list[dict[str, Any]], method: str = "first_is_agent") -> list[dict[str, Any]]:
        speech_indices = [i for i, s in enumerate(segments) if s.get("speaker") != "silence"]
        if not speech_indices:
            return segments

        if method == "f0_based":
            # If f0-based IDs are already present we keep them, else keep existing speaker_A/B.
            pass
        elif method != "first_is_agent":
            method = "first_is_agent"

        speaker_order: list[str] = []
        for idx in speech_indices:
            spk = str(segments[idx].get("speaker", "speaker_A"))
            if spk not in speaker_order:
                speaker_order.append(spk)

        if not speaker_order:
            return segments

        if method == "first_is_agent":
            role_map = {speaker_order[0]: "agent"}
            if len(speaker_order) > 1:
                role_map[speaker_order[1]] = "customer"
        else:
            # fallback
            role_map = {speaker_order[0]: "agent"}
            if len(speaker_order) > 1:
                role_map[speaker_order[1]] = "customer"

        for seg in segments:
            spk = seg.get("speaker")
            if spk == "silence":
                seg["role"] = "silence"
            elif spk in role_map:
                seg["role"] = role_map[spk]
            else:
                seg["role"] = "customer"
        return segments

    def _segments_from_mask(
        self,
        speech_mask: np.ndarray,
        rms: np.ndarray,
        sample_rate: int,
        frame_samples: int,
        signal: np.ndarray,
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        if len(speech_mask) == 0:
            return segments

        start = 0
        current = bool(speech_mask[0])
        for i in range(1, len(speech_mask)):
            state = bool(speech_mask[i])
            if state != current:
                segments.append(self._make_segment(start, i, current, rms, sample_rate, frame_samples, signal))
                start = i
                current = state
        segments.append(self._make_segment(start, len(speech_mask), current, rms, sample_rate, frame_samples, signal))

        # Merge tiny speech segments into neighboring silence if below min duration.
        filtered: list[dict[str, Any]] = []
        for seg in segments:
            dur = float(seg["end"]) - float(seg["start"])
            if seg["speaker"] != "silence" and dur < self.min_segment_duration:
                seg["speaker"] = "silence"
                seg.pop("f0_mean", None)
            if filtered and filtered[-1]["speaker"] == seg["speaker"]:
                filtered[-1]["end"] = seg["end"]
                filtered[-1]["energy"] = round((filtered[-1]["energy"] + seg["energy"]) / 2, 6)
                if seg["speaker"] != "silence":
                    f0_a = filtered[-1].get("f0_mean", 0.0)
                    f0_b = seg.get("f0_mean", 0.0)
                    filtered[-1]["f0_mean"] = round((f0_a + f0_b) / 2, 3)
            else:
                filtered.append(seg)

        speech_idxs = [i for i, s in enumerate(filtered) if s["speaker"] != "silence"]
        self._assign_speaker_ids(filtered, speech_idxs)
        return filtered

    def _assign_speaker_ids(self, segments: list[dict[str, Any]], speech_idxs: list[int]) -> None:
        if not speech_idxs:
            return

        f0_values = [segments[i].get("f0_mean", 0.0) for i in speech_idxs]
        has_f0 = any(v > 0 for v in f0_values)
        labels: Optional[list[int]] = None

        if has_f0:
            try:
                from sklearn.cluster import KMeans

                x = np.array([[v] for v in f0_values], dtype=np.float32)
                model = KMeans(n_clusters=2, n_init=10, random_state=42)
                labels = model.fit_predict(x).tolist()
            except Exception:
                labels = None

        if labels is None:
            # Fallback: alternating turns.
            labels = [idx % 2 for idx in range(len(speech_idxs))]

        for seg_idx, lab in zip(speech_idxs, labels):
            segments[seg_idx]["speaker"] = "speaker_A" if lab == 0 else "speaker_B"

    def _make_segment(
        self,
        frame_start: int,
        frame_end: int,
        is_speech: bool,
        rms: np.ndarray,
        sample_rate: int,
        frame_samples: int,
        signal: np.ndarray,
    ) -> dict[str, Any]:
        start_sec = (frame_start * frame_samples) / sample_rate
        end_sec = (frame_end * frame_samples) / sample_rate
        energy = float(np.mean(rms[frame_start:frame_end])) if frame_end > frame_start else 0.0

        seg: dict[str, Any] = {
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "speaker": "speech" if is_speech else "silence",
            "energy": round(energy, 6),
        }

        if is_speech:
            s0 = frame_start * frame_samples
            s1 = frame_end * frame_samples
            seg_signal = signal[s0:s1]
            seg["f0_mean"] = round(self._estimate_f0(seg_signal, sample_rate), 3)
        return seg

    @staticmethod
    def _estimate_f0(signal: np.ndarray, sample_rate: int) -> float:
        if signal.size < int(sample_rate * 0.1):
            return 0.0
        try:
            import librosa

            f0, _, _ = librosa.pyin(signal, sr=sample_rate, fmin=65, fmax=400)
            if f0 is None:
                return 0.0
            voiced = f0[~np.isnan(f0)]
            if voiced.size == 0:
                return 0.0
            return float(np.mean(voiced))
        except Exception:
            return 0.0

    @staticmethod
    def _load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        try:
            return torchaudio.load(str(path))
        except Exception:
            try:
                import soundfile as sf

                data, sr = sf.read(str(path), always_2d=True, dtype="float32")
                data = torch.from_numpy(data.T)
                return data, int(sr)
            except Exception as exc:
                raise RuntimeError(f"Failed to load audio for diarization: {path}") from exc


class StreamDiarizer:
    """Lightweight Diarization fuer Streaming-Kontext."""

    def __init__(self):
        self.speakers: dict[str, dict[str, float]] = {}
        self.chunk_count = 0

    def identify(self, audio_chunk: np.ndarray) -> str:
        self.chunk_count += 1
        if audio_chunk.size == 0:
            return "silence"

        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2) + 1e-9))
        if rms < 0.008:
            return "silence"

        f0 = self._estimate_f0_fast(audio_chunk, sample_rate=16000)
        zcr = self._zcr(audio_chunk)
        overlap_hint = self._overlap_hint(audio_chunk, rms, zcr)
        if overlap_hint:
            self._update_profile("agent", f0, rms)
            self._update_profile("customer", f0 * 1.05 if f0 > 0 else 0.0, rms)
            return "both"

        if not self.speakers:
            self._update_profile("agent", f0, rms)
            return "agent"

        if "customer" not in self.speakers and "agent" in self.speakers:
            agent_f0 = self.speakers["agent"].get("f0", 0.0)
            if f0 > 0 and agent_f0 > 0 and abs(f0 - agent_f0) > 25 and self.chunk_count >= 2:
                self._update_profile("customer", f0, rms)
                return "customer"
            self._update_profile("agent", f0, rms)
            return "agent"

        if "agent" not in self.speakers:
            self._update_profile("agent", f0, rms)
            return "agent"
        if "customer" not in self.speakers:
            self._update_profile("customer", f0, rms)
            return "customer"

        agent_f0 = self.speakers["agent"].get("f0", 0.0)
        customer_f0 = self.speakers["customer"].get("f0", 0.0)
        if f0 <= 0:
            # Bei unvoiced Segmenten naechsten RMS-Profilpunkt nutzen.
            a_rms = self.speakers["agent"].get("rms", rms)
            c_rms = self.speakers["customer"].get("rms", rms)
            role = "agent" if abs(rms - a_rms) <= abs(rms - c_rms) else "customer"
            self._update_profile(role, f0, rms)
            return role

        role = "agent" if abs(f0 - agent_f0) <= abs(f0 - customer_f0) else "customer"
        self._update_profile(role, f0, rms)
        return role

    def _update_profile(self, role: str, f0: float, rms: float) -> None:
        state = self.speakers.get(role, {"f0": f0 if f0 > 0 else 0.0, "rms": rms})
        if f0 > 0:
            state["f0"] = 0.7 * state.get("f0", f0) + 0.3 * f0
        state["rms"] = 0.7 * state.get("rms", rms) + 0.3 * rms
        self.speakers[role] = state

    @staticmethod
    def _overlap_hint(audio_chunk: np.ndarray, rms: float, zcr: float) -> bool:
        peak = float(np.max(np.abs(audio_chunk))) if audio_chunk.size else 0.0
        return rms > 0.06 and zcr > 0.15 and peak > 0.5

    @staticmethod
    def _zcr(audio_chunk: np.ndarray) -> float:
        x = audio_chunk.astype(np.float32)
        if x.size < 2:
            return 0.0
        signs = np.signbit(x)
        changes = np.sum(signs[1:] != signs[:-1])
        return float(changes) / float(x.size - 1)

    @staticmethod
    def _estimate_f0_fast(audio_chunk: np.ndarray, sample_rate: int) -> float:
        x = audio_chunk.astype(np.float32)
        if x.size < int(0.05 * sample_rate):
            return 0.0
        x = x - float(np.mean(x))
        if np.max(np.abs(x)) < 1e-4:
            return 0.0
        x = x / (np.max(np.abs(x)) + 1e-8)
        min_lag = max(1, int(sample_rate / 400))
        max_lag = max(min_lag + 1, int(sample_rate / 65))
        if x.size <= max_lag:
            return 0.0
        ac = np.correlate(x, x, mode="full")[x.size - 1 :]
        search = ac[min_lag:max_lag]
        if search.size == 0:
            return 0.0
        lag = int(np.argmax(search)) + min_lag
        val = float(search[lag - min_lag])
        if val < 0.05:
            return 0.0
        return float(sample_rate) / float(lag)
