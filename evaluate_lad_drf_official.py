import os
import csv
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score, f1_score as f1
from tqdm import tqdm

from dataset import MVSADataset
from dataset_robust import RobustMVSADataset
from disruptions import DisruptionStrategy, DisruptionType
from models.lad_drf import LADDRF

# ==================== 配置 ====================
data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
test_json = os.path.join(data_root, "test.json")
model_path = "best_lad_drf_c.pth"  # 主模型
results_dir = "results/lad_drf"
MASK_RATIO_THR = 0.5
TEXT_GATE_THRESHOLD = 0.35  # 与训练一致即可；主路由不依赖它
SEEDS = [42, 43, 44]  # 可改为 [42,43,44,45,46]
RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
# ==============================================


def mask_ratio(input_ids, tokenizer):
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    not_pad = (input_ids != pad_id).float()
    n_mask = (input_ids == mask_id).float().sum(dim=1)
    n_tok = not_pad.sum(dim=1).clamp(min=1.0)
    return n_mask / n_tok


def is_empty_text(input_ids, tokenizer):
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id
    special = {pad_id, cls_id, sep_id, mask_id}
    flags = []
    for i in range(input_ids.size(0)):
        cnt = sum(1 for t in input_ids[i].tolist() if t not in special)
        flags.append(cnt <= 1)
    return torch.tensor(flags, device=input_ids.device)


def rule_use_image(input_ids, tokenizer):
    """与 evaluate_lad_drf_rule_route 一致"""
    ratio = mask_ratio(input_ids, tokenizer)
    empty = is_empty_text(input_ids, tokenizer)
    return (ratio >= MASK_RATIO_THR) | empty


@torch.no_grad()
def run_eval(model, loader, device, tokenizer):
    """规则路由评估；同时返回 Acc / W-F1 / Macro-F1"""
    preds, labels = [], []
    n_img, n_tot = 0, 0
    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        b = y.size(0)
        n_tot += b

        use_img = rule_use_image(input_ids, tokenizer)
        n_img += use_img.sum().item()

        logits = model(
            images,
            input_ids,
            attention_mask,
            update_queue=False,
            force_image_path=use_img.float(),
        )
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        labels.extend(y.cpu().tolist())

    acc = accuracy_score(labels, preds)
    w_f1 = f1_score(labels, preds, average="weighted")
    m_f1 = f1_score(labels, preds, average="macro")
    return acc, w_f1, m_f1, n_img / max(n_tot, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Official eval (规则路由) | {model_path}")
    print(f"MASK_RATIO_THR={MASK_RATIO_THR} | Random seeds={SEEDS}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = LADDRF(
        num_classes=3,
        text_gate_threshold=TEXT_GATE_THRESHOLD,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rows = []

    # ----- Clean -----
    clean_loader = DataLoader(
        MVSADataset(test_json, image_dir, tokenizer, is_train=False),
        batch_size=16, shuffle=False, num_workers=0,
    )
    acc, w, m, img_r = run_eval(model, clean_loader, device, tokenizer)
    print(f"\n[clean] Acc={acc:.4f} W-F1={w:.4f} Macro-F1={m:.4f} | ImgPath={img_r:.1%}")
    rows.append({
        "mode": "clean", "modality": "-", "type": "-", "ratio": 0,
        "acc": round(acc, 4), "w_f1": round(w, 4), "macro_f1": round(m, 4),
        "acc_std": "", "w_f1_std": "", "img_path_ratio": round(img_r, 4),
    })

    # ----- Fixed -----
    print("\n----- Fixed -----")
    print(f"{'Modality':<8} {'Type':<4} {'Acc':>8} {'W-F1':>8} {'Macro':>8} {'Img%':>8}")
    for modality in ["image", "text"]:
        for dtype in [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]:
            ds = RobustMVSADataset(
                test_json, image_dir, tokenizer, is_train=False,
                strategy=DisruptionStrategy.FIXED,
                disruption_type=dtype,
                disruption_ratio=1.0,
                disrupted_modality=modality,
                seed=42,
            )
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
            acc, w, m, img_r = run_eval(model, loader, device, tokenizer)
            print(
                f"{modality:<8} {dtype.value:<4} {acc:8.4f} {w:8.4f} {m:8.4f} {img_r*100:7.1f}%"
            )
            rows.append({
                "mode": "fixed", "modality": modality, "type": dtype.value, "ratio": 1.0,
                "acc": round(acc, 4), "w_f1": round(w, 4), "macro_f1": round(m, 4),
                "acc_std": "", "w_f1_std": "", "img_path_ratio": round(img_r, 4),
            })

    # ----- Random -----
    print("\n----- Random (mean±std) -----")
    for ratio in RATIOS:
        for dtype in [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]:
            accs, ws, ms = [], [], []
            for seed in SEEDS:
                ds = RobustMVSADataset(
                    test_json, image_dir, tokenizer, is_train=False,
                    strategy=DisruptionStrategy.RANDOM,
                    disruption_type=dtype,
                    disruption_ratio=ratio,
                    disrupted_modality="modality",
                    seed=seed,
                )
                loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
                acc, w, m, _ = run_eval(model, loader, device, tokenizer)
                accs.append(acc)
                ws.append(w)
                ms.append(m)
            am, astd = float(np.mean(accs)), float(np.std(accs))
            wm, wstd = float(np.mean(ws)), float(np.std(ws))
            mm = float(np.mean(ms))
            print(
                f"[random] dr={ratio:.1f} {dtype.value:3s} "
                f"Acc={am:.4f}±{astd:.4f} W-F1={wm:.4f}±{wstd:.4f}"
            )
            rows.append({
                "mode": "random", "modality": "random", "type": dtype.value, "ratio": ratio,
                "acc": round(am, 4), "w_f1": round(wm, 4), "macro_f1": round(mm, 4),
                "acc_std": round(astd, 4), "w_f1_std": round(wstd, 4),
                "img_path_ratio": "",
            })

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"lad_official_{ts}.csv")
    fields = [
        "mode", "modality", "type", "ratio",
        "acc", "w_f1", "macro_f1", "acc_std", "w_f1_std", "img_path_ratio",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已保存: {csv_path}")
    print("可直接用于论文表格（规则路由 + Clean/Fixed/Random）")


if __name__ == "__main__":
    main()