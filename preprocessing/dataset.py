"""
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

LABEL2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_MAP.values())}
ID2LABEL = {idx: emotion for emotion, idx in LABEL2ID.items()}


def parse_ravdess_filename(file_path: str) -> Optional[Dict[str, any]]:
    filename = Path(file_path).stem
    parts = filename.split("-")

    emotion_code = parts[2]
    actor_id = int(parts[6])

    if emotion_code not in EMOTION_MAP:
        return None

    emotion_label = EMOTION_MAP[emotion_code]

    return {
        "path": str(file_path),
        "emotion": emotion_label,
        "label": LABEL2ID[emotion_label],
        "actor": actor_id,
        "gender": "female" if actor_id % 2 == 0 else "male",
    }


def create_speaker_split(
    data_dir: str,
    train_actors: List[int] = list(range(1, 19)),     # Actors 01 - 18
    val_actors: List[int] = list(range(19, 21)),       # Actors 19 - 20
    test_actors: List[int] = list(range(21, 25)),      # Actors 21 - 24
):
    """
    Splits the data into Train, Val, and Test by Actor ID.
    """
    audio_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    
    train_samples = []
    val_samples = []
    test_samples = []

    for file_path in audio_files:
        meta = parse_ravdess_filename(file_path)
        if meta is None:
            continue

        actor = meta["actor"]
        if actor in train_actors:
            train_samples.append(meta)
        elif actor in val_actors:
            val_samples.append(meta)
        elif actor in test_actors:
            test_samples.append(meta)

    return train_samples, val_samples, test_samples


class RAVDESSDataset(Dataset):
    """
    PyTorch Dataset for RAVDESS audio samples compatible with Wav2Vec 2.0.
    """
    def __init__(
        self,
        samples: List[Dict],
        feature_extractor: Optional[AutoFeatureExtractor] = None,
        target_sampling_rate: int = 16000,
        max_duration_sec: float = 3.5,
    ):
        self.samples = samples
        self.feature_extractor = feature_extractor
        self.target_sampling_rate = target_sampling_rate
        self.max_length = int(target_sampling_rate * max_duration_sec)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        file_path = item["path"]
        label = item["label"]

        # 1. Load Audio
        waveform, sample_rate = torchaudio.load(file_path)

        # 2. Convert Stereo to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Resample to 16,000 Hz if needed
        if sample_rate != self.target_sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.target_sampling_rate
            )
            waveform = resampler(waveform)

        waveform = waveform.squeeze()

        # 4. Feature Extraction (Wav2Vec2 normalization & padding/truncation)
        if self.feature_extractor is not None:
            inputs = self.feature_extractor(
                waveform.numpy(),
                sampling_rate=self.target_sampling_rate,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_values = inputs.input_values.squeeze(0)
        else:
            # Fallback manual padding/truncating if not using HF Feature Extractor
            if waveform.shape[0] > self.max_length:
                input_values = waveform[:self.max_length]
            else:
                padding = self.max_length - waveform.shape[0]
                input_values = torch.nn.functional.pad(waveform, (0, padding))

        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }