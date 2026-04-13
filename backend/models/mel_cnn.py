import torch
import torch.nn as nn
import torchaudio


class MelCNN(nn.Module):
    """Simple CNN on log-mel spectrograms as a fast baseline."""

    def __init__(self, n_mels: int = 80, n_classes: int = 2):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=160, n_mels=n_mels
        )
        self.instance_norm = nn.InstanceNorm2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(x)
        mel = torch.log(mel + 1e-9)
        mel = mel.unsqueeze(1)
        mel = self.instance_norm(mel)
        features = self.conv(mel)
        return self.classifier(features)
