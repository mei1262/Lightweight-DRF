import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import MVSADataset
from dataset_robust import RobustMVSADataset
from disruptions import DisruptionStrategy, DisruptionType
from models.lad_drf import LADDRF

data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
test_json = os.path.join(data_root, "test.json")
model_path = "best_lad_drf.pth"
TEXT_GATE_THRESHOLD = 0.35


@torch.no_grad()
def evaluate(model, loader, device):
    preds, labels = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        logits = model(
            batch["image"].to(device),
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            update_queue=False,
            force_image_path=None,
        )
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
    return accuracy_score(labels, preds), f1_score(labels, preds, average="weighted")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"LAD-DRF | {model_path} | threshold={TEXT_GATE_THRESHOLD}\n")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = LADDRF(num_classes=3, text_gate_threshold=TEXT_GATE_THRESHOLD).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # clean
    clean_loader = DataLoader(
        MVSADataset(test_json, image_dir, tokenizer, is_train=False),
        batch_size=16, shuffle=False, num_workers=0,
    )
    acc, w = evaluate(model, clean_loader, device)
    print(f"[clean] Acc={acc:.4f} | W-F1={w:.4f}\n")

    # fixed
    print(f"{'Modality':<8} {'Type':<4} {'Acc':>8} {'W-F1':>8}")
    print("-" * 32)
    for modality in ["image", "text"]:
        for dtype in [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]:
            ds = RobustMVSADataset(
                json_path=test_json,
                image_dir=image_dir,
                tokenizer=tokenizer,
                is_train=False,
                strategy=DisruptionStrategy.FIXED,
                disruption_type=dtype,
                disruption_ratio=1.0,
                disrupted_modality=modality,
                seed=42,
            )
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
            acc, w = evaluate(model, loader, device)
            print(f"{modality:<8} {dtype.value:<4} {acc:8.4f} {w:8.4f}")

    print("\n参考:")
    print("  Baseline     clean=0.7257  Image-D=0.6637  Text-D=0.6482")
    print("  no_recovery  clean=0.7301  Image-D=0.7146  Text-D=0.5951")


if __name__ == "__main__":
    main()