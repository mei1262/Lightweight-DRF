import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset_robust import RobustMVSADataset
from disruptions import DisruptionStrategy, DisruptionType
from models.lq_drf_v2 import LQDRFv2

data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
test_json = os.path.join(data_root, "test.json")
model_path = "best_lq_drf_v2.pth"


@torch.no_grad()
def evaluate(model, loader, device):
    preds, labels = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        logits = model(
            batch["image"].to(device),
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            update_queue=False,
        )
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
    acc = accuracy_score(labels, preds)
    w_f1 = f1_score(labels, preds, average="weighted")
    return acc, w_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: LQ-DRF-v2 | ckpt={model_path}\n")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = LQDRFv2(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    modalities = ["image", "text"]
    types = [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]

    print(f"{'Modality':<8} {'Type':<4} {'Acc':>8} {'W-F1':>8}")
    print("-" * 32)

    for modality in modalities:
        for dtype in types:
            dataset = RobustMVSADataset(
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
            loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
            acc, w_f1 = evaluate(model, loader, device)
            print(f"{modality:<8} {dtype.value:<4} {acc:8.4f} {w_f1:8.4f}")

    print("\n对比参考:")
    print("  Baseline  Image-D=0.6637  Text-D=0.6482")
    print("  no_recov  Image-D=0.7146  Text-D=0.5951")
    print("  full旧    Image-D=0.6969  Text-D=0.5951")


if __name__ == "__main__":
    main()