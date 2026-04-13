"""Whisper-based transcription with segment/word timestamps."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf


class Transcriber:
    """Whisper-basierte Transkription mit Wort-Timestamps."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None
        self._stream_buffer = np.asarray([], dtype=np.float32)
        self._stream_offset = 0.0
        self._window_seconds = 2.0
        self._load_model()

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        except Exception:
            self._model = None

    def is_available(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_path: str) -> list:
        if self._model is None:
            return []
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        segments, _ = self._model.transcribe(
            str(path),
            word_timestamps=True,
            vad_filter=True,
        )
        return self._to_dict_segments(segments, offset=0.0)

    def transcribe_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict:
        """Streaming-Version: buffer 2 Sekunden und transkribiert dann."""
        if self._model is None:
            return {"ready": False, "text": "", "segments": []}

        chunk = np.asarray(audio_chunk, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        self._stream_buffer = np.concatenate([self._stream_buffer, chunk])

        required = int(self._window_seconds * sr)
        if len(self._stream_buffer) < required:
            return {"ready": False, "text": "", "segments": []}

        window = self._stream_buffer[:required]
        self._stream_buffer = self._stream_buffer[required:]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(str(tmp_path), window, sr, subtype="PCM_16")
            segments, _ = self._model.transcribe(
                str(tmp_path),
                word_timestamps=True,
                vad_filter=True,
            )
            segs = self._to_dict_segments(segments, offset=self._stream_offset)
            text = " ".join(s["text"] for s in segs if s.get("text")).strip()
            out = {"ready": True, "text": text, "segments": segs}
            self._stream_offset += self._window_seconds
            return out
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _to_dict_segments(segments: object, offset: float) -> list:
        out: list[dict] = []
        for seg in segments:
            words = []
            raw_words = getattr(seg, "words", None) or []
            for w in raw_words:
                words.append(
                    {
                        "word": str(getattr(w, "word", "")).strip(),
                        "start": round(float(getattr(w, "start", 0.0) or 0.0) + offset, 3),
                        "end": round(float(getattr(w, "end", 0.0) or 0.0) + offset, 3),
                    }
                )
            out.append(
                {
                    "start": round(float(getattr(seg, "start", 0.0) or 0.0) + offset, 3),
                    "end": round(float(getattr(seg, "end", 0.0) or 0.0) + offset, 3),
                    "text": str(getattr(seg, "text", "") or "").strip(),
                    "words": words,
                }
            )
        return out
