import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset_robust import RobustMVSADataset
from disruptions import DisruptionStrategy, DisruptionType
from models.lad_drf import LADDRF

# ==================== 路径 ====================
data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
test_json = os.path.join(data_root, "test.json")
model_path = "best_lad_drf.pth"
TEXT_GATE_THRESHOLD = 0.35
# ==============================================

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


@torch.no_grad()
def run_one(model, loader, device, force_image=False, collect_quality=False):
    """
    force_image=True 时：全部样本强制走图像头
    collect_quality=True 时：收集 txt_quality 统计（仅 auto 路由有意义）
    """
    model.eval()
    all_preds, all_labels = [], []
    all_txt_q = []
    n_low_quality = 0
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
            # 自动路由；额外算一次 quality 便于统计
            img_feat, txt_feat, img_q, txt_q = model.encode(
                images, input_ids, attention_mask, update_queue=False
            )
            if collect_quality:
                all_txt_q.append(txt_q.cpu().numpy())
                n_low_quality += (txt_q < model.text_gate_threshold).sum().item()

            logits = model(
                images, input_ids, attention_mask,
                update_queue=False, force_image_path=None,
            )

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    w_f1 = f1_score(all_labels, all_preds, average="weighted")
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    bc_pred = np.bincount(preds_arr, minlength=3)
    bc_true = np.bincount(labels_arr, minlength=3)

    info = {
        "acc": acc,
        "w_f1": w_f1,
        "bc_pred": bc_pred,
        "bc_true": bc_true,
        "n_total": n_total,
        "n_low_quality": n_low_quality,
        "txt_q_mean": None,
        "txt_q_std": None,
        "frac_low_q": None,
    }
    if collect_quality and all_txt_q:
        q = np.concatenate(all_txt_q)
        info["txt_q_mean"] = float(q.mean())
        info["txt_q_std"] = float(q.std())
        info["frac_low_q"] = n_low_quality / max(n_total, 1)
    return info


def print_info(title, info):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Acc = {info['acc']:.4f}  |  W-F1 = {info['w_f1']:.4f}")
    print(f"样本数 N = {info['n_total']}")
    print(f"真实标签分布 [neg, neu, pos] = {info['bc_true'].tolist()}")
    print(f"预测标签分布 [neg, neu, pos] = {info['bc_pred'].tolist()}")
    # 多数类基线
    maj = info["bc_true"].argmax()
    maj_acc = info["bc_true"][maj] / info["n_total"]
    print(f"多数类({LABEL_NAMES[maj]})瞎猜 Acc ≈ {maj_acc:.4f}")
    if info["txt_q_mean"] is not None:
        print(f"txt_quality: mean={info['txt_q_mean']:.4f} std={info['txt_q_std']:.4f}")
        print(
            f"txt_quality < {TEXT_GATE_THRESHOLD} 的比例 = {info['frac_low_q']:.4f} "
            f"({info['n_low_quality']}/{info['n_total']})"
        )


def make_loader(tokenizer, modality, dtype):
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
    print(f"threshold = {TEXT_GATE_THRESHOLD}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = LADDRF(num_classes=3, text_gate_threshold=TEXT_GATE_THRESHOLD).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ----- 重点：Fixed Text-D -----
    loader_td = make_loader(tokenizer, "text", DisruptionType.D)

    info_auto = run_one(
        model, loader_td, device, force_image=False, collect_quality=True
    )
    print_info("Text-D | 自动路由 (当前 LAD-DRF)", info_auto)

    info_force = run_one(
        model, loader_td, device, force_image=True, collect_quality=False
    )
    print_info("Text-D | 强制全部走图像头", info_force)

    # ----- 对照：Fixed Image-D 自动路由 -----
    loader_id = make_loader(tokenizer, "image", DisruptionType.D)
    info_img = run_one(
        model, loader_id, device, force_image=False, collect_quality=True
    )
    print_info("Image-D | 自动路由 (对照)", info_img)

    # ----- 诊断结论提示 -----
    print("\n" + "=" * 60)
    print("诊断提示")
    print("=" * 60)
    acc_a, acc_f = info_auto["acc"], info_force["acc"]
    print(f"自动路由 Text-D Acc = {acc_a:.4f}")
    print(f"强制图像 Text-D Acc = {acc_f:.4f}")
    if acc_f - acc_a > 0.02:
        print("→ 强制明显更好：问题多半在【路由/阈值】，下一步降低阈值或加 MASK 触发。")
    elif abs(acc_f - acc_a) <= 0.02 and acc_f < 0.62:
        print("→ 强制也差不多且仍低：问题多半在【图像头没学好】，下一步加大 FORCE_IMAGE_PROB 重训。")
    else:
        print("→ 看预测分布：若几乎全是 pos，则是多数类塌缩。")

    if info_auto["frac_low_q"] is not None:
        if info_auto["frac_low_q"] < 0.2:
            print(
                f"→ Text-D 时低质量比例只有 {info_auto['frac_low_q']:.2%}，"
                "路由可能几乎没打开。"
            )
        else:
            print(
                f"→ Text-D 时低质量比例 {info_auto['frac_low_q']:.2%}，"
                "路由有打开，若 Acc 仍低则图像头本身偏弱。"
            )


if __name__ == "__main__":
    main()