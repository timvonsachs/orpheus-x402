"""Runtime-Konfiguration fuer Orpheus Acoustic Sense."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    orpheus_api_url: str = os.getenv("ORPHEUS_API_URL", "http://localhost:8000/api/v1/analyze")
    model_checkpoint: str = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_mel_cnn.pt")
    device: str = os.getenv("DEVICE", "cpu")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8001"))
    version: str = os.getenv("ORPHEUS_VERSION", "0.1.0")


settings = Settings()
