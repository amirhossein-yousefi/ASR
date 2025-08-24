from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import random
import numpy as np
import soundfile as sf
import librosa
import torch


@dataclass
class AugmentConfig:
    speed_prob: float = 0.3
    gain_prob: float = 0.3
    noise_prob: float = 0.3
    speed_min: float = 0.9
    speed_max: float = 1.1
    gain_min_db: float = -6.0
    gain_max_db: float = 6.0
    snr_min_db: float = 5.0
    snr_max_db: float = 25.0


class WaveformAugment:
    """
    Lightweight waveform augmentation (CPU friendly):
    - Speed perturbation via resampling
    - Random gain
    - Random noise mix-in at target SNR (if noise clips available)
    """
    def __init__(self, cfg: AugmentConfig, noise_dir: Optional[str], sample_rate: int) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.noise_paths: List[str] = []
        if noise_dir and os.path.isdir(noise_dir):
            for root, _, files in os.walk(noise_dir):
                for f in files:
                    if f.lower().endswith((".wav", ".flac", ".ogg")):
                        self.noise_paths.append(os.path.join(root, f))

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        # Speed perturb
        if random.random() < self.cfg.speed_prob:
            factor = random.uniform(self.cfg.speed_min, self.cfg.speed_max)
            target_len = int(round(len(wav) / factor))
            wav = librosa.resample(wav, orig_sr=self.sample_rate, target_sr=int(self.sample_rate * factor))
            # Resample back to original sr and length
            wav = librosa.resample(wav, orig_sr=int(self.sample_rate * factor), target_sr=self.sample_rate)
            wav = librosa.util.fix_length(wav, target_len)

        # Gain
        if random.random() < self.cfg.gain_prob:
            gain_db = random.uniform(self.cfg.gain_min_db, self.cfg.gain_max_db)
            wav = wav * (10 ** (gain_db / 20.0))

        # Noise
        if self.noise_paths and random.random() < self.cfg.noise_prob:
            noise_path = random.choice(self.noise_paths)
            noise, nsr = sf.read(noise_path)
            if noise.ndim > 1:
                noise = noise.mean(axis=1)
            if nsr != self.sample_rate:
                noise = librosa.resample(noise, orig_sr=nsr, target_sr=self.sample_rate)

            if len(noise) < len(wav):
                # loop noise if too short
                reps = int(np.ceil(len(wav) / len(noise)))
                noise = np.tile(noise, reps)
            noise = noise[: len(wav)]

            # Mix at target SNR
            snr_db = random.uniform(self.cfg.snr_min_db, self.cfg.snr_max_db)
            wav_power = np.mean(wav ** 2) + 1e-9
            noise_power = np.mean(noise ** 2) + 1e-9
            desired_noise_power = wav_power / (10 ** (snr_db / 10.0))
            noise = noise * np.sqrt(desired_noise_power / noise_power)
            wav = wav + noise

        # Avoid clipping
        max_abs = np.max(np.abs(wav)) + 1e-9
        if max_abs > 1.0:
            wav = wav / max_abs
        return wav
