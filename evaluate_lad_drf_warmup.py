import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import MVSADataset
from dataset_robust import RobustMVSADataset
from disruptions import DisruptionStrategy, DisruptionType
from models.lad_drf import LADDRF

# ==================== 路径 ====================
data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
train_json = os.path.join(data_root, "train.json")
test_json = os.path.join(data_root, "test.json")
model_path = "best_lad_drf.pth"
TEXT_GATE_THRESHOLD = 0.35
# 预热用多少个 train batch（填满队列即可，队列默认 512）
WARMUP_BATCHES = 40
# ==============================================


@torch.no_grad()
def warmup_queues(model, loader, device, max_batches=40):
    """只用训练集前向，更新队列；不训练、不反传。"""
    model.train()  # 允许 update_queue 写入
    n = 0
    for batch in tqdm(loader, desc="Warmup queues", leave=False):
        if n >= max_batches:
            break
        model.encode(
            batch["image"].to(device),
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            update_queue=True,
        )
        n += 1
    model.eval()
    img_len = len(model.img_queue.queue)
    txt_len = len(model.txt_queue.queue)
    print(f"队列预热完成: img_queue={img_len}, txt_queue={txt_len}")
    img_mean, img_std = model.img_queue.get_mean_std()
    txt_mean, txt_std = model.txt_queue.get_mean_std()
    print(f"img mean/std ready: {img_mean is not None}, txt mean/std ready: {txt_mean is not None}")


@torch.no_grad()
def evaluate(model, loader, device, force_image=False, collect_quality=False):
    model.eval()
    all_preds, all_labels = [], []
    all_txt_q = []
    n_low = 0
    n_total = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        b = labels.size(0)
        n_total += b

        if force_image:
            force = torch.ones(b, device=device)
            logits = model(
                images, input_ids, attention_mask,
                update_queue=False, force_image_path=force,
            )
        else:
            if collect_quality:
                _, _, _, txt_q = model.encode(
                    images, input_ids, attention_mask, update_queue=False
                )
                all_txt_q.append(txt_q.cpu().numpy())
                n_low += (txt_q < model.text_gate_threshold).sum().item()

            logits = model(
                images, input_ids, attention_mask,
                update_queue=False, force_image_path=None,
            )

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    w_f1 = f1_score(all_labels, all_preds, average="weighted")
    bc_pred = np.bincount(np.array(all_preds), minlength=3)

    q_mean = q_std = frac_low = None
    if all_txt_q:
        q = np.concatenate(all_txt_q)
        q_mean, q_std = float(q.mean()), float(q.std())
        frac_low = n_low / max(n_total, 1)

    return {
        "acc": acc,
        "w_f1": w_f1,
        "bc_pred": bc_pred,
        "n_total": n_total,
        "n_low": n_low,
        "q_mean": q_mean,
        "q_std": q_std,
        "frac_low": frac_low,
    }


def print_result(title, info):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Acc={info['acc']:.4f} | W-F1={info['w_f1']:.4f}")
    print(f"预测分布 [neg,neu,pos]={info['bc_pred'].tolist()}")
    if info["q_mean"] is not None:
        print(
            f"txt_q mean={info['q_mean']:.4f} std={info['q_std']:.4f} | "
            f"q<{TEXT_GATE_THRESHOLD} 比例={info['frac_low']:.4f} ({info['n_low']}/{info['n_total']})"
        )


def make_fixed_loader(tokenizer, modality, dtype):
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
    return DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ckpt: {model_path}")
    print(f"threshold={TEXT_GATE_THRESHOLD} | warmup_batches={WARMUP_BATCHES}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = LADDRF(num_classes=3, text_gate_threshold=TEXT_GATE_THRESHOLD).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # ----- 1) 用训练集预热队列 -----
    train_loader = DataLoader(
        MVSADataset(train_json, image_dir, tokenizer, is_train=False),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    warmup_queues(model, train_loader, device, max_batches=WARMUP_BATCHES)

    # ----- 2) Clean -----
    clean_loader = DataLoader(
        MVSADataset(test_json, image_dir, tokenizer, is_train=False),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    info_clean = evaluate(model, clean_loader, device, force_image=False, collect_quality=True)
    print_result("Clean | 自动路由 + 预热队列", info_clean)

    # ----- 3) Fixed 全表 -----
    print("\n----- Fixed disruption (warmup) -----")
    print(f"{'Modality':<8} {'Type':<4} {'Acc':>8} {'W-F1':>8}")
    print("-" * 32)
    for modality in ["image", "text"]:
        for dtype in [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]:
            loader = make_fixed_loader(tokenizer, modality, dtype)
            info = evaluate(model, loader, device, force_image=False, collect_quality=False)
            print(f"{modality:<8} {dtype.value:<4} {info['acc']:8.4f} {info['w_f1']:8.4f}")

    # ----- 4) Text-D 详细：自动 vs 强制图像 -----
    loader_td = make_fixed_loader(tokenizer, "text", DisruptionType.D)

    info_auto = evaluate(model, loader_td, device, force_image=False, collect_quality=True)
    print_result("Text-D | 自动路由 + 预热队列", info_auto)

    info_force = evaluate(model, loader_td, device, force_image=True, collect_quality=False)
    print_result("Text-D | 强制图像头 (对照)", info_force)

    print("\n参考: no_recovery clean=0.7301 Image-D=0.7146 Text-D=0.5951")
    print("参考: 强制图像头此前 Text-D≈0.6527")


if __name__ == "__main__":
    main()