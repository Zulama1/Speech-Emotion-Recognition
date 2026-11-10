# Speech Emotion Recognition (SER) via Hybrid CRNN

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchaudio](https://img.shields.io/badge/torchaudio-Compatible-orange?logo=python&logoColor=white)](https://pytorch.org/audio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end deep learning framework for **Speech Emotion Recognition (SER)** built on the **RAVDESS** dataset. The model extracts 2D Log-Mel Spectrogram representations from standardized audio signals and models both spatial acoustic patterns and temporal sequences using a hybrid **Convolutional Recurrent Neural Network (CRNN: 2D CNN + BiLSTM)**.

Crucially, this benchmark uses an academic **speaker-independent evaluation split**, ensuring the model generalizes to unseen speakers rather than memorizing individual vocal signatures.

---
## Key Features

- **Lightweight Audio Pipeline:** Uses `soundfile` and native `torchaudio.transforms` directly for resampling and spectrogram calculation, eliminating heavy dependencies like `librosa` and avoiding Windows FFmpeg/`torchcodec` runtime incompatibilities.
- **Rigorous Speaker-Independent Splitting:** Prevents vocal identity leakage across splits by strictly isolating distinct actor cohorts:
  - **Train:** Actors 01–18 (1,080 clips)
  - **Validation:** Actors 19–20 (120 clips)
  - **Held-Out Test:** Actors 21–24 (240 clips)
- **Hybrid CRNN Architecture:**
  - **Spatial Feature Extractor:** 3-layer 2D CNN with Batch Normalization, ReLU activations, Max Pooling, and Dropout.
  - **Temporal Context Modeling:** 2-layer Bidirectional LSTM (BiLSTM) capturing prosodic dynamics across the utterance duration.
  - **Utterance Pooling & Classification Head:** Temporal mean-pooling over LSTM outputs into a regularized dense classification head across 8 emotion classes.
- **Reproducible Evaluation:** Comprehensive metrics generation including class-wise precision, recall, F1-scores, and confusion matrices.

---

## Dataset: RAVDESS

The model is trained on the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**:
- 1,440 speech audio samples spoken by 24 professional actors (12 male, 12 female).
- **8 Emotional Categories:**
  1. `neutral`
  2. `calm`
  3. `happy`
  4. `sad`
  5. `angry`
  6. `fearful`
  7. `disgust`
  8. `surprised`

Filename naming structure follows:
```text
03-01-[Emotion]-[Intensity]-[Statement]-[Repetition]-[Actor].wav
```

---

## Project Structure

```text
DL-LAB/
├── Data/
│   └── raw/
│       ├── Actor_01/
│       ├── Actor_02/
│       └── ... (Actor_01 to Actor_24)
├── preprocessing/
│   ├── __init__.py
│   ├── dataset.py          # Speaker-aware splitting and label metadata
│   └── features.py         # Audio loading, resampling, Log-Mel extraction
├── src/
│   ├── __init__.py
│   ├── model.py            # Hybrid 2D CNN + BiLSTM architecture
│   ├── train.py            # Training loop with validation checkpointing
│   ├── evaluate.py         # Held-out speaker evaluation & metric reports
│   └── best_model.pt       # Saved best validation checkpoint
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/speech-emotion-recognition-crnn.git
cd speech-emotion-recognition-crnn
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional for GPU acceleration)*: If an NVIDIA GPU is available, install CUDA-enabled PyTorch wheels:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Dataset Setup

1. Download the audio dataset from [Kaggle RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).
2. Extract the archive into `Data/raw/` so each actor resides in their respective directory:
   ```text
   Data/raw/Actor_01/03-01-01-01-01-01-01.wav
   Data/raw/Actor_02/...
   ```

---

## Usage

### 1. Train the Model
The training script dynamically samples, normalizes, and extracts 128-band Log-Mel Spectrograms on-the-fly and saves the top-performing checkpoint to `src/best_model.pt`:
```bash
python -m src.train
```

### 2. Evaluate on Unseen Test Speakers
Run inference on the held-out test cohort (Actors 21–24):
```bash
python -m src.evaluate
```

---

## Benchmark Results (Actors 21–24)

Under strict speaker-independent partitioning, the model achieves **50.00% overall accuracy** on the held-out test cohort, outperforming the random chance baseline (12.50%) by a factor of 4.

### Classification Report

| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Neutral** | 0.35 | 0.38 | 0.36 | 16 |
| **Calm** | 0.53 | 0.88 | 0.66 | 32 |
| **Happy** | 0.23 | 0.25 | 0.24 | 32 |
| **Sad** | 0.26 | 0.16 | 0.20 | 32 |
| **Angry** | 0.57 | 0.53 | 0.55 | 32 |
| **Fearful** | 0.44 | 0.25 | 0.32 | 32 |
| **Disgust** | 0.69 | 0.84 | 0.76 | 32 |
| **Surprised** | 0.76 | 0.69 | 0.72 | 32 |
| **Accuracy** | | | **0.50** | **240** |
| **Macro Avg** | 0.48 | 0.50 | 0.48 | 240 |
| **Weighted Avg** | 0.49 | 0.50 | 0.48 | 240 |

### Confusion Matrix

```text
               Predicted Class
         Neu  Cal  Hap  Sad  Ang  Fea  Dis  Sur
Neutral [  6    5    2    3    0    0    0    0 ]
Calm    [  2   28    0    2    0    0    0    0 ]
Happy   [  5    1    8    5    3    8    0    2 ]
Sad     [  0   16    5    5    0    0    6    0 ]
Angry   [  0    0    7    0   17    2    2    4 ]
Fearful [  4    2   10    4    0    8    3    1 ]
Disgust [  0    1    3    0    1    0   27    0 ]
Surprised[ 0    0    0    0    9    0    1   22 ]
```

### Acoustic Error Analysis
1. **High Arousal Separation:** Emotions characterized by distinct vocal energy envelopes and pitch dynamics (*Disgust* $F_1 = 0.76$, *Surprised* $F_1 = 0.72$) show robust generalization.
2. **Low Arousal Misclassifications:** 16 out of 32 *Sad* instances are classified as *Calm*, reflecting acoustic similarities in depressed fundamental pitch ($F_0$) contours and low vocal energy.
3. **High Intensity Crossovers:** Acoustic strain in high-energy states leads to cross-confusion between *Surprised* and *Angry* (9 instances), as well as *Fearful* and *Happy* (10 instances).

---

## Future Directions
- **Self-Supervised Pretraining:** Fine-tuning transformer backbones like `wav2vec2-base` or `WavLM` to replace handcrafted spectrogram extraction.
- **Multimodal Architectures:** Incorporating automatic speech recognition (ASR) transcripts to disambiguate acoustically flat emotions (e.g., *Sad* vs. *Calm*) using linguistic semantic embeddings.

---

## License
Distributed under the MIT License. See `LICENSE` for more information.