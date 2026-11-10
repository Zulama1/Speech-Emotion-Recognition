import os
from typing import Tuple, Optional
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf

def load_and_resample_audio(
    file_path: str, 
    target_sr: int = 16000, 
    to_mono: bool = True
):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Read audio using soundfile
    data, sr = sf.read(file_path, dtype="float32")
    waveform = torch.from_numpy(data)

    # Standardize dimensions to (channels, time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()

    # Convert to mono
    if to_mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if sample rate does not match target_sr
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # CRITICAL: waveform MUST be first, sr MUST be second
    return waveform.squeeze(0), sr


def pad_or_truncate(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Pads or truncates a 1D waveform to a fixed number of samples.
    """
    if waveform.shape[0] > target_length:
        return waveform[:target_length]
    elif waveform.shape[0] < target_length:
        padding = target_length - waveform.shape[0]
        return torch.nn.functional.pad(waveform, (0, padding))
    return waveform


def extract_log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    win_length: Optional[int] = None,
    hop_length: int = 512,
    n_mels: int = 128,
) -> torch.Tensor:
    # Validate type to catch misordered arguments early
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"Expected waveform to be a torch.Tensor, got {type(waveform)}")

    # Ensure waveform is at least 2D: (channel, time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

    mel_spec = mel_transform(waveform)
    log_mel_spec = amplitude_to_db(mel_spec)

    # Return shape: (n_mels, time_steps)
    return log_mel_spec.squeeze(0)

def compute_zcr(waveform: torch.Tensor, frame_length: int = 1024, hop_length: int = 512) -> torch.Tensor:
    """
    Computes Zero-Crossing Rate using native PyTorch operations.
    """
    # Pad to center frames
    pad = frame_length // 2
    padded = torch.nn.functional.pad(waveform, (pad, pad), mode="reflect")
    
    # Unfold into sliding frames: (num_frames, frame_length)
    frames = padded.unfold(dimension=0, size=frame_length, step=hop_length)
    
    # Compute sign changes across adjacent points
    signs = torch.sign(frames)
    signs[signs == 0] = 1
    zcr = torch.mean(torch.abs(signs[:, 1:] - signs[:, :-1]) / 2.0, dim=1)
    return zcr


def compute_rms(waveform: torch.Tensor, frame_length: int = 1024, hop_length: int = 512) -> torch.Tensor:
    """ 
    Computes RMS Energy.
    """
    pad = frame_length // 2
    padded = torch.nn.functional.pad(waveform, (pad, pad), mode="reflect")
    frames = padded.unfold(dimension=0, size=frame_length, step=hop_length)
    return torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-8)


def extract_classical_features(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Extracts statistical 1D summary features (Mel-log graph, ZCR, RMS).
    """
    waveform, sr = load_and_resample_audio(file_path, target_sr=target_sr)


    #Mel Spectrogram
    mel = extract_log_mel_spectrogram(waveform, sample_rate=sr, n_mels=64)
    mel_mean = torch.mean(mel, dim=1).numpy()
    mel_std = torch.std(mel, dim=1).numpy()

    #Zero-Crossing Rate
    zcr = compute_zcr(waveform)
    zcr_mean = np.array([torch.mean(zcr).item()])
    zcr_std = np.array([torch.std(zcr).item()])

    #RMS Energy
    rms = compute_rms(waveform)
    rms_mean = np.array([torch.mean(rms).item()])
    rms_std = np.array([torch.std(rms).item()])

    #1D feature vector
    return np.hstack([
        mel_mean, mel_std,
        zcr_mean, zcr_std,
        rms_mean, rms_std
    ])