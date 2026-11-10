import sys
import os
from pathlib import Path
import torchaudio.transforms as T

# Add project root to sys.path to resolve imports cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from preprocessing.dataset import create_speaker_split
from preprocessing.features import load_and_resample_audio, pad_or_truncate, extract_log_mel_spectrogram
from src.model import EmotionCRNN


import torchaudio.transforms as T

class SpectrogramDataset(Dataset):
    """Loads audio on the fly, extracts Log-Mel Spectrograms, and applies optional augmentation."""
    def __init__(self, samples, target_sr=16000, duration_sec=3.5, n_mels=128, is_train=False):
        self.samples = samples
        self.target_sr = target_sr
        self.target_length = int(target_sr * duration_sec)
        self.n_mels = n_mels
        self.is_train = is_train

        # SpecAugment transforms (only applied when is_train=True)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Unpack exactly as (waveform, sr)
        waveform, sr = load_and_resample_audio(item["path"], target_sr=self.target_sr)
        
        # 2. Pad or truncate
        waveform = pad_or_truncate(waveform, self.target_length)
        
        # 3. Extract Spectrogram
        spec = extract_log_mel_spectrogram(
            waveform=waveform,
            sample_rate=sr,
            n_mels=self.n_mels
        )

        # 4. Standardize (Zero mean, unit variance per sample)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = spec.unsqueeze(0)  # Shape: (1, n_mels, time_steps)

        # DO NOT apply freq_mask or time_mask here!
        return spec, torch.tensor(item["label"], dtype=torch.long)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    raw_data_dir = ROOT_DIR / "Data" / "raw"
    train_meta, val_meta, _ = create_speaker_split(str(raw_data_dir))

    train_ds = SpectrogramDataset(train_meta, is_train=True)
    val_ds = SpectrogramDataset(val_meta, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = EmotionCRNN(num_classes=8, n_mels=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 30
    best_val_acc = 0.0
    save_path = ROOT_DIR / "src" / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * specs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            total_train += specs.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * specs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                total_val += specs.size(0)

        val_acc = val_correct / total_val if total_val > 0 else 0
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss/total_train:.4f} Acc: {train_correct/total_train:.2%} | Val Loss: {val_loss/total_val:.4f} Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path.name}")


if __name__ == "__main__":
    train()