import os
import json
import csv
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset_robust import RobustMVSADataset
from disruptions import DisruptionStrategy, DisruptionType
from models.lightweight_drf_ablation import LightweightDRFAblation

# ==================== 路径配置 ====================
data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
test_json = os.path.join(data_root, "test.json")
results_dir = "results/ablation_robust"
# ==================================================

# 消融模型配置: name -> (use_quality_weight, use_recovery, checkpoint)
ABLATIONS = {
    "full":        (True,  True,  "best_ablation_full.pth"),
    "no_recovery": (True,  False, "best_ablation_no_recovery.pth"),
    "no_quality":  (False, True,  "best_ablation_no_quality.pth"),
    "no_both":     (False, False, "best_ablation_no_both.pth"),
}


def build_model(name, device):
    use_qw, use_rec, ckpt = ABLATIONS[name]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"找不到模型文件: {ckpt}")

    model = LightweightDRFAblation(
        num_classes=3,
        use_quality_weight=use_qw,
        use_recovery=use_rec,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def evaluate_loader(model, loader, device):
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(images, input_ids, attention_mask, update_queue=False)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    w_f1 = f1_score(all_labels, all_preds, average="weighted")
    m_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, w_f1, m_f1


def run_fixed(models, tokenizer, device, batch_size=16, seed=42):
    """Fixed 干扰：单次即可（确定性强）"""
    rows = []
    modalities = ["image", "text"]
    types = [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]

    for name in models:
        model = build_model(name, device)
        print(f"\n===== Fixed | {name} =====")

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
                    seed=seed,
                )
                loader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=0
                )
                acc, w_f1, m_f1 = evaluate_loader(model, loader, device)

                row = {
                    "mode": "fixed",
                    "model": name,
                    "modality": modality,
                    "type": dtype.value,
                    "ratio": 1.0,
                    "acc": round(acc, 4),
                    "weighted_f1": round(w_f1, 4),
                    "macro_f1": round(m_f1, 4),
                    "acc_mean": "",
                    "acc_std": "",
                    "weighted_f1_mean": "",
                    "weighted_f1_std": "",
                    "macro_f1_mean": "",
                    "macro_f1_std": "",
                    "seeds": str(seed),
                }
                rows.append(row)
                print(
                    f"[fixed] {name:12s} | {modality:5s} | {dtype.value:3s} | "
                    f"Acc={acc:.4f} | W-F1={w_f1:.4f}"
                )

    return rows


def run_random(
    models,
    tokenizer,
    device,
    ratios=None,
    seeds=None,
    batch_size=16,
):
    """Random 干扰：多种子重复，报均值和标准差"""
    if ratios is None:
        ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]  # 5个种子；时间紧可改成 [42, 43, 44]

    rows = []
    types = [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]

    for name in models:
        model = build_model(name, device)
        print(f"\n===== Random | {name} | seeds={seeds} =====")

        for ratio in ratios:
            for dtype in types:
                accs, wf1s, mf1s = [], [], []

                for seed in seeds:
                    dataset = RobustMVSADataset(
                        json_path=test_json,
                        image_dir=image_dir,
                        tokenizer=tokenizer,
                        is_train=False,
                        strategy=DisruptionStrategy.RANDOM,
                        disruption_type=dtype,
                        disruption_ratio=ratio,
                        disrupted_modality="image",  # random 模式下实际随机选择
                        seed=seed,
                    )
                    loader = DataLoader(
                        dataset, batch_size=batch_size, shuffle=False, num_workers=0
                    )
                    acc, w_f1, m_f1 = evaluate_loader(model, loader, device)
                    accs.append(acc)
                    wf1s.append(w_f1)
                    mf1s.append(m_f1)

                acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
                w_mean, w_std = float(np.mean(wf1s)), float(np.std(wf1s))
                m_mean, m_std = float(np.mean(mf1s)), float(np.std(mf1s))

                row = {
                    "mode": "random",
                    "model": name,
                    "modality": "random",
                    "type": dtype.value,
                    "ratio": ratio,
                    "acc": "",
                    "weighted_f1": "",
                    "macro_f1": "",
                    "acc_mean": round(acc_mean, 4),
                    "acc_std": round(acc_std, 4),
                    "weighted_f1_mean": round(w_mean, 4),
                    "weighted_f1_std": round(w_std, 4),
                    "macro_f1_mean": round(m_mean, 4),
                    "macro_f1_std": round(m_std, 4),
                    "seeds": str(seeds),
                }
                rows.append(row)
                print(
                    f"[random] {name:12s} | dr={ratio:.1f} | {dtype.value:3s} | "
                    f"Acc={acc_mean:.4f}±{acc_std:.4f} | "
                    f"W-F1={w_mean:.4f}±{w_std:.4f}"
                )

    return rows


def save_results(rows, prefix="ablation_robust"):
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"{prefix}_{ts}.json")
    csv_path = os.path.join(results_dir, f"{prefix}_{ts}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fields = [
        "mode",
        "model",
        "modality",
        "type",
        "ratio",
        "acc",
        "weighted_f1",
        "macro_f1",
        "acc_mean",
        "acc_std",
        "weighted_f1_mean",
        "weighted_f1_std",
        "macro_f1_mean",
        "macro_f1_std",
        "seeds",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n已保存 JSON: {json_path}")
    print(f"已保存 CSV : {csv_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model_names = list(ABLATIONS.keys())

    all_rows = []

    # 1) Fixed 干扰（单 seed）
    print("\n" + "=" * 60)
    print("开始 Fixed 干扰评估")
    print("=" * 60)
    all_rows.extend(run_fixed(model_names, tokenizer, device))

    # 2) Random 干扰（多种子，报 mean±std）
    print("\n" + "=" * 60)
    print("开始 Random 干扰评估（多种子）")
    print("=" * 60)
    # 时间紧可改成 seeds=[42, 43, 44]
    all_rows.extend(
        run_random(
            model_names,
            tokenizer,
            device,
            ratios=[0.2, 0.4, 0.6, 0.8, 1.0],
            seeds=[42, 43, 44, 45, 46],
        )
    )

    save_results(all_rows)
    print("\n消融 Fixed + Random 评估全部完成。")


if __name__ == "__main__":
    main()