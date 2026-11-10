import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing.dataset import create_speaker_split, ID2LABEL
from src.model import EmotionCRNN
from src.train import SpectrogramDataset


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data_dir = ROOT_DIR / "Data" / "raw"
    model_path = ROOT_DIR / "src" / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError("Trained weights not found. Run train.py first.")

    # Load speaker-independent test split
    _, _, test_meta = create_speaker_split(str(raw_data_dir))
    test_ds = SpectrogramDataset(test_meta)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Load Model
    model = EmotionCRNN(num_classes=8, n_mels=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            outputs = model(specs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]

    print("\n--- Test Set Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=target_names))

    print("--- Confusion Matrix ---")
    print(confusion_matrix(all_targets, all_preds))


if __name__ == "__main__":
    evaluate()