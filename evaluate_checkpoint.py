import argparse
import os
import csv
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import MVSADataset
from dataset_robust import RobustMVSADataset
from disruptions import DisruptionStrategy, DisruptionType
from models.lightweight_drf_ablation import LightweightDRFAblation
from models.baseline import BaselineModel

data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
test_json = os.path.join(data_root, "test.json")

# 与 LAD official 对齐
SEEDS = [42, 43, 44]
RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
# 与 LAD 相同：若 dataset 不支持 random，改成 "image"
RANDOM_MODALITY = "random"
RESULTS_DIR = "results/checkpoint_eval"


@torch.no_grad()
def run_eval(model, loader, device):
    preds, labels = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)

        if isinstance(model, BaselineModel):
            logits = model(images, input_ids, attention_mask)
        else:
            logits = model(images, input_ids, attention_mask, update_queue=False)

        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        labels.extend(label.cpu().tolist())

    acc = accuracy_score(labels, preds)
    w_f1 = f1_score(labels, preds, average="weighted")
    m_f1 = f1_score(labels, preds, average="macro")
    return acc, w_f1, m_f1


def build_model(args, device):
    if args.baseline:
        model = BaselineModel(num_classes=3).to(device)
    else:
        model = LightweightDRFAblation(
            num_classes=3,
            use_quality_weight=bool(args.quality),
            use_recovery=bool(args.recovery),
        ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    return model


def eval_clean(model, tokenizer, device):
    loader = DataLoader(
        MVSADataset(test_json, image_dir, tokenizer, is_train=False),
        batch_size=16, shuffle=False, num_workers=0,
    )
    acc, w_f1, m_f1 = run_eval(model, loader, device)
    print(f"[clean] Acc={acc:.4f} | W-F1={w_f1:.4f} | Macro-F1={m_f1:.4f}")
    return acc, w_f1, m_f1


def eval_fixed(model, tokenizer, device):
    print(f"{'Modality':<8} {'Type':<4} {'Acc':>8} {'W-F1':>8} {'Macro':>8}")
    print("-" * 40)
    rows = []
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
            acc, w_f1, m_f1 = run_eval(model, loader, device)
            print(f"{modality:<8} {dtype.value:<4} {acc:8.4f} {w_f1:8.4f} {m_f1:8.4f}")
            rows.append({
                "mode": "fixed", "modality": modality, "type": dtype.value,
                "ratio": 1.0, "acc": acc, "w_f1": w_f1, "macro_f1": m_f1,
                "acc_std": "", "w_f1_std": "",
            })
    return rows


def eval_random(model, tokenizer, device):
    print(f"Random modality={RANDOM_MODALITY} | seeds={SEEDS}")
    print(f"{'ratio':<6} {'Type':<4} {'Acc':>18} {'W-F1':>18}")
    print("-" * 50)
    rows = []
    for ratio in RATIOS:
        for dtype in [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]:
            accs, ws, ms = [], [], []
            for seed in SEEDS:
                ds = RobustMVSADataset(
                    json_path=test_json,
                    image_dir=image_dir,
                    tokenizer=tokenizer,
                    is_train=False,
                    strategy=DisruptionStrategy.RANDOM,
                    disruption_type=dtype,
                    disruption_ratio=ratio,
                    disrupted_modality=RANDOM_MODALITY,
                    seed=seed,
                )
                loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
                acc, w_f1, m_f1 = run_eval(model, loader, device)
                accs.append(acc)
                ws.append(w_f1)
                ms.append(m_f1)
            am, astd = float(np.mean(accs)), float(np.std(accs))
            wm, wstd = float(np.mean(ws)), float(np.std(ws))
            mm = float(np.mean(ms))
            print(f"{ratio:<6.1f} {dtype.value:<4} {am:.4f}±{astd:.4f}   {wm:.4f}±{wstd:.4f}")
            rows.append({
                "mode": "random", "modality": "random", "type": dtype.value,
                "ratio": ratio, "acc": round(am, 4), "w_f1": round(wm, 4),
                "macro_f1": round(mm, 4),
                "acc_std": round(astd, 4), "w_f1_std": round(wstd, 4),
            })
    return rows


def save_rows(name, rows):
    if not rows:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{name}_{ts}.csv")
    fields = ["mode", "modality", "type", "ratio", "acc", "w_f1", "macro_f1", "acc_std", "w_f1_std"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已保存: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--name", type=str, default="model")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--quality", type=int, default=1, choices=[0, 1])
    parser.add_argument("--recovery", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["clean", "fixed", "random", "all"],
        help="clean | fixed | random | all",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"找不到权重: {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Name: {args.name} | mode={args.mode}")
    print(f"Ckpt: {args.ckpt}")
    if not args.baseline:
        print(f"Structure: quality={bool(args.quality)} recovery={bool(args.recovery)}")
    print()

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = build_model(args, device)
    print("权重加载成功\n")

    rows = []
    if args.mode in ("clean", "all"):
        eval_clean(model, tokenizer, device)
        print()
    if args.mode in ("fixed", "all"):
        rows.extend(eval_fixed(model, tokenizer, device))
        print()
    if args.mode in ("random", "all"):
        rows.extend(eval_random(model, tokenizer, device))

    if args.mode in ("fixed", "random", "all"):
        save_rows(args.name, rows)


if __name__ == "__main__":
    main()