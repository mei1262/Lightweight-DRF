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
test_json = os.path.join(data_root, "test.json")
#model_path = "best_lad_drf_a.pth"
#model_path = "best_lad_drf_b.pth"
model_path = "best_lad_drf_c.pth"
TEXT_GATE_THRESHOLD = 0.35  # 模型内仍保留，本脚本路由不依赖它
# 规则：MASK 占比超过该值 → 走图像头
MASK_RATIO_THR = 0.5
# ==============================================


def mask_ratio(input_ids, tokenizer):
    """每个样本 [MASK] 占非 pad token 的比例, 返回 [B] tensor"""
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    ids = input_ids
    not_pad = (ids != pad_id).float()
    n_mask = (ids == mask_id).float().sum(dim=1)
    n_tok = not_pad.sum(dim=1).clamp(min=1.0)
    return n_mask / n_tok


def is_empty_text(input_ids, attention_mask, tokenizer):
    """有效 token 很少（几乎没有真实词）→ 视为缺失"""
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id
    # 非特殊、非 pad 的 token 数
    special = {pad_id, cls_id, sep_id, mask_id}
    b, l = input_ids.shape
    flags = []
    for i in range(b):
        cnt = 0
        for t in input_ids[i].tolist():
            if t not in special:
                cnt += 1
        flags.append(cnt <= 1)  # 几乎没有实词
    return torch.tensor(flags, device=input_ids.device)


@torch.no_grad()
def evaluate_rule(model, loader, device, tokenizer, force_all_image=False):
    model.eval()
    all_preds, all_labels = [], []
    n_img_path = 0
    n_total = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        b = labels.size(0)
        n_total += b

        if force_all_image:
            use_img = torch.ones(b, dtype=torch.bool, device=device)
        else:
            ratio = mask_ratio(input_ids, tokenizer)
            empty = is_empty_text(input_ids, attention_mask, tokenizer)
            use_img = (ratio >= MASK_RATIO_THR) | empty

        n_img_path += use_img.sum().item()
        logits = model(
            images, input_ids, attention_mask,
            update_queue=False,
            force_image_path=use_img.float(),
        )
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    w_f1 = f1_score(all_labels, all_preds, average="weighted")
    bc = np.bincount(np.array(all_preds), minlength=3)
    return {
        "acc": acc,
        "w_f1": w_f1,
        "bc_pred": bc,
        "frac_img_path": n_img_path / max(n_total, 1),
        "n_img_path": n_img_path,
        "n_total": n_total,
    }


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
    print(f"规则路由: MASK占比>={MASK_RATIO_THR} 或 几乎无实词 → 图像头\n")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = LADDRF(num_classes=3, text_gate_threshold=TEXT_GATE_THRESHOLD).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 1) Clean
    clean_loader = DataLoader(
        MVSADataset(test_json, image_dir, tokenizer, is_train=False),
        batch_size=16, shuffle=False, num_workers=0,
    )
    info = evaluate_rule(model, clean_loader, device, tokenizer)
    print("=" * 60)
    print("Clean | 规则路由")
    print("=" * 60)
    print(f"Acc={info['acc']:.4f} | W-F1={info['w_f1']:.4f}")
    print(f"走图像头比例={info['frac_img_path']:.4f} ({info['n_img_path']}/{info['n_total']})")
    print(f"预测分布[neg,neu,pos]={info['bc_pred'].tolist()}")

    # 2) Fixed 全表
    print("\n----- Fixed | 规则路由 -----")
    print(f"{'Modality':<8} {'Type':<4} {'Acc':>8} {'W-F1':>8} {'ImgPath%':>10}")
    print("-" * 44)
    for modality in ["image", "text"]:
        for dtype in [DisruptionType.C, DisruptionType.D, DisruptionType.C_D]:
            loader = make_fixed_loader(tokenizer, modality, dtype)
            info = evaluate_rule(model, loader, device, tokenizer)
            print(
                f"{modality:<8} {dtype.value:<4} {info['acc']:8.4f} {info['w_f1']:8.4f} "
                f"{info['frac_img_path']*100:9.1f}%"
            )

    # 3) Text-D 对照：规则 vs 强制全图像
    loader_td = make_fixed_loader(tokenizer, "text", DisruptionType.D)
    info_rule = evaluate_rule(model, loader_td, device, tokenizer, force_all_image=False)
    info_force = evaluate_rule(model, loader_td, device, tokenizer, force_all_image=True)

    print("\n" + "=" * 60)
    print("Text-D 对照")
    print("=" * 60)
    print(f"规则路由   Acc={info_rule['acc']:.4f}  图像头比例={info_rule['frac_img_path']:.2%}")
    print(f"强制全图像 Acc={info_force['acc']:.4f}  图像头比例={info_force['frac_img_path']:.2%}")
    print("\n参考: 质量自动路由 Text-D=0.5951 | 强制图像≈0.65~0.66 | Baseline Text-D=0.6482")


if __name__ == "__main__":
    main()