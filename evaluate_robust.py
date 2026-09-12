"""
Robustness evaluation for Baseline and Lightweight DRF.

Usage examples
--------------
# 1) Modality-fixed (matches paper Table 1 style, no retraining required)
python evaluate_robust.py --mode fixed --models baseline drf

# 2) Modality-random at one disruption ratio
python evaluate_robust.py --mode random --disruption-ratio 0.6 --models baseline drf

# 3) Full random sweep (dr = 0.2, 0.4, 0.6, 0.8, 1.0)
python evaluate_robust.py --mode random --full-sweep --models baseline drf
"""

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer

from dataset_robust import RobustMVSADataset
from disruptions import DisruptionStrategy, DisruptionType
from models.baseline import BaselineModel
from models.lightweight_drf import LightweightDRF

# ==================== Default paths ====================
DATA_ROOT = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
IMAGE_DIR = os.path.join(DATA_ROOT, "data")
TEST_JSON = os.path.join(DATA_ROOT, "test.json")
BASELINE_CKPT = "best_baseline.pth"
DRF_CKPT = "best_lightweight_drf.pth"
RESULTS_DIR = "results/robustness"
# =======================================================


def build_model(model_name: str, device: torch.device):
    if model_name == "baseline":
        model = BaselineModel(num_classes=3).to(device)
        ckpt = BASELINE_CKPT
    elif model_name == "drf":
        model = LightweightDRF(num_classes=3, queue_size=512).to(device)
        ckpt = DRF_CKPT
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_loader(
    model,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> Tuple[float, float, float]:
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        if model_name == "drf":
            logits = model(images, input_ids, attention_mask, update_queue=False)
        else:
            logits = model(images, input_ids, attention_mask)

        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, weighted_f1, macro_f1


def run_single_setting(
    model,
    model_name: str,
    tokenizer,
    device: torch.device,
    strategy: DisruptionStrategy,
    disruption_type: DisruptionType,
    disrupted_modality: Optional[str],
    disruption_ratio: float,
    batch_size: int,
    seed: int,
) -> Dict[str, float]:
    dataset = RobustMVSADataset(
        json_path=TEST_JSON,
        image_dir=IMAGE_DIR,
        tokenizer=tokenizer,
        is_train=False,
        strategy=strategy,
        disruption_type=disruption_type,
        disruption_ratio=disruption_ratio,
        disrupted_modality=disrupted_modality or "image",
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    acc, weighted_f1, macro_f1 = evaluate_loader(model, loader, device, model_name)
    return {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
    }


def run_fixed_experiments(
    models: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> List[Dict]:
    rows: List[Dict] = []
    modalities = ["image", "text"]
    disruption_types = [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]

    for model_name in models:
        model, ckpt = build_model(model_name, device)
        print(f"\n===== Fixed disruption | {model_name} | ckpt={ckpt} =====")

        for modality in modalities:
            for disruption_type in disruption_types:
                metrics = run_single_setting(
                    model=model,
                    model_name=model_name,
                    tokenizer=tokenizer,
                    device=device,
                    strategy=DisruptionStrategy.FIXED,
                    disruption_type=disruption_type,
                    disrupted_modality=modality,
                    disruption_ratio=1.0,
                    batch_size=batch_size,
                    seed=seed,
                )
                row = {
                    "mode": "fixed",
                    "model": model_name,
                    "disrupted_modality": modality,
                    "disruption_type": disruption_type.value,
                    "disruption_ratio": 1.0,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"[fixed] {model_name} | {modality} | {disruption_type.value} | "
                    f"Acc={metrics['accuracy']:.4f} | W-F1={metrics['weighted_f1']:.4f}"
                )

    return rows


def run_random_experiments(
    models: List[str],
    tokenizer,
    device: torch.device,
    disruption_ratios: List[float],
    batch_size: int,
    seed: int,
) -> List[Dict]:
    rows: List[Dict] = []
    disruption_types = [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]

    for model_name in models:
        model, ckpt = build_model(model_name, device)
        print(f"\n===== Random disruption | {model_name} | ckpt={ckpt} =====")
        print(
            "Note: paper retrains with random disruption; "
            "this script evaluates already-trained checkpoints (inference-only)."
        )

        for dr in disruption_ratios:
            for disruption_type in disruption_types:
                metrics = run_single_setting(
                    model=model,
                    model_name=model_name,
                    tokenizer=tokenizer,
                    device=device,
                    strategy=DisruptionStrategy.RANDOM,
                    disruption_type=disruption_type,
                    disrupted_modality=None,
                    disruption_ratio=dr,
                    batch_size=batch_size,
                    seed=seed,
                )
                row = {
                    "mode": "random",
                    "model": model_name,
                    "disrupted_modality": "random",
                    "disruption_type": disruption_type.value,
                    "disruption_ratio": dr,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"[random] {model_name} | dr={dr:.1f} | {disruption_type.value} | "
                    f"Acc={metrics['accuracy']:.4f} | W-F1={metrics['weighted_f1']:.4f}"
                )

    return rows


def run_clean_baseline(
    models: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> List[Dict]:
    rows: List[Dict] = []
    for model_name in models:
        model, ckpt = build_model(model_name, device)
        dataset = RobustMVSADataset(
            json_path=TEST_JSON,
            image_dir=IMAGE_DIR,
            tokenizer=tokenizer,
            is_train=False,
            strategy=None,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        acc, weighted_f1, macro_f1 = evaluate_loader(model, loader, device, model_name)
        rows.append(
            {
                "mode": "clean",
                "model": model_name,
                "disrupted_modality": "none",
                "disruption_type": "none",
                "disruption_ratio": 0.0,
                "accuracy": acc,
                "weighted_f1": weighted_f1,
                "macro_f1": macro_f1,
            }
        )
        print(
            f"[clean] {model_name} | Acc={acc:.4f} | W-F1={weighted_f1:.4f} | ckpt={ckpt}"
        )
    return rows


def save_results(rows: List[Dict], output_prefix: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(RESULTS_DIR, f"{output_prefix}_{timestamp}.json")
    csv_path = os.path.join(RESULTS_DIR, f"{output_prefix}_{timestamp}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "mode",
        "model",
        "disrupted_modality",
        "disruption_type",
        "disruption_ratio",
        "accuracy",
        "weighted_f1",
        "macro_f1",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Robustness evaluation for MVSA models")
    parser.add_argument(
        "--mode",
        choices=["clean", "fixed", "random", "all"],
        default="fixed",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline", "drf"],
        default=["baseline", "drf"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--disruption-ratio",
        type=float,
        default=1.0,
        help="Random-mode disruption ratio (0.2~1.0)",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Random mode: sweep dr in {0.2,0.4,0.6,0.8,1.0}",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", type=str, default="robust_eval")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    all_rows: List[Dict] = []

    if args.mode in ("clean", "all"):
        all_rows.extend(
            run_clean_baseline(args.models, tokenizer, device, args.batch_size)
        )

    if args.mode in ("fixed", "all"):
        all_rows.extend(
            run_fixed_experiments(
                args.models, tokenizer, device, args.batch_size, args.seed
            )
        )

    if args.mode in ("random", "all"):
        ratios = [0.2, 0.4, 0.6, 0.8, 1.0] if args.full_sweep else [args.disruption_ratio]
        all_rows.extend(
            run_random_experiments(
                args.models, tokenizer, device, ratios, args.batch_size, args.seed
            )
        )

    save_results(all_rows, args.output_prefix)


if __name__ == "__main__":
    main()
