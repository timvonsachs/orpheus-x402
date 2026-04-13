"""MelCNN-based humanness scoring."""

from __future__ import annotations

from typing import Any

import torch
import torchaudio

from models.mel_cnn import MelCNN


class HumannessEngine:
    def __init__(self, checkpoint_path: str, device: str = "cpu", max_duration: float = 4.0):
        self.device = device
        self.sample_rate = 16000
        self.max_duration = max_duration
        self.model = MelCNN().to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def score_audio(self, waveform: torch.Tensor, sr: int) -> dict[str, Any]:
        waveform = self._prepare_waveform(waveform, sr)
        max_samples = int(self.max_duration * self.sample_rate)

        if waveform.shape[0] <= max_samples:
            return self._score_chunk(waveform)

        scores = []
        for start in range(0, waveform.shape[0], max_samples):
            chunk = waveform[start : start + max_samples]
            if chunk.numel() == 0:
                continue
            scores.append(self._score_chunk(chunk))

        avg_score = sum(s["score"] for s in scores) / max(len(scores), 1)
        avg_conf = sum(s["confidence"] for s in scores) / max(len(scores), 1)
        classification = "human" if avg_score >= 50 else "synthetic"
        return {
            "score": round(avg_score, 1),
            "confidence": round(avg_conf, 3),
            "classification": classification,
            "interpretation": self._interpretation(avg_score),
            "chunk_scores": scores,
        }

    def score_segments(
        self, waveform: torch.Tensor, sr: int, segment_duration: float = 5.0
    ) -> list[dict[str, Any]]:
        waveform = self._prepare_waveform(waveform, sr)
        segment_samples = int(max(segment_duration, 0.5) * self.sample_rate)
        if segment_samples <= 0:
            return []

        segments = []
        for start in range(0, waveform.shape[0], segment_samples):
            chunk = waveform[start : start + segment_samples]
            if chunk.numel() == 0:
                continue
            result = self._score_chunk(chunk)
            segments.append(
                {
                    "start": round(start / self.sample_rate, 3),
                    "end": round(min(start + segment_samples, waveform.shape[0]) / self.sample_rate, 3),
                    "humanness_score": result["score"],
                }
            )
        return segments

    def _prepare_waveform(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        return waveform

    def _score_chunk(self, waveform: torch.Tensor) -> dict[str, Any]:
        max_samples = int(self.max_duration * self.sample_rate)
        if waveform.shape[0] < max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[0]))
        elif waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]

        with torch.no_grad():
            logits = self.model(waveform.unsqueeze(0).to(self.device))
            probs = torch.softmax(logits, dim=1)[0]
            score = probs[1].item() * 100
            confidence = max(probs[0].item(), probs[1].item())

        classification = "human" if score >= 50 else "synthetic"
        return {
            "score": round(score, 1),
            "confidence": round(confidence, 3),
            "classification": classification,
            "interpretation": self._interpretation(score),
        }

    @staticmethod
    def _interpretation(score: float) -> str:
        if score >= 70:
            return "Audio klingt ueberwiegend natuerlich mit ausreichender prosodischer Varianz."
        if score >= 50:
            return "Audio liegt im Grenzbereich zwischen natuerlicher und synthetischer Sprachcharakteristik."
        if score >= 30:
            return "Audio ist wahrscheinlich TTS-generiert. Hinweise: reduzierte Mikro-Variation und gleichfoermige Dynamik."
        return "Audio ist sehr wahrscheinlich synthetisch mit stark regelmaessiger Prosodie."
