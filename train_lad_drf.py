import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import MVSADataset
from models.lad_drf import LADDRF

# ==================== 路径 ====================
data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
train_json = os.path.join(data_root, "train.json")
val_json = os.path.join(data_root, "val.json")
# ==============================================

# 训练时以该概率强制走图像头（模拟文本失效）
FORCE_IMAGE_PROB = 0.45#0.350.55
TEXT_GATE_THRESHOLD = 0.35
SAVE_PATH = "best_lad_drf_c.pth" # 新文件，别覆盖 a

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        b = labels.size(0)
        # 每个样本是否强制图像路径
        force_img = torch.rand(b, device=device) < FORCE_IMAGE_PROB

        optimizer.zero_grad()
        logits = model(
            images,
            input_ids,
            attention_mask,
            update_queue=True,
            force_image_path=force_img,
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_fuse, labels_all = [], []
    preds_img = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        b = labels.size(0)

        # A) 默认路径（干净 val 上基本是融合）
        logits_fuse = model(
            images, input_ids, attention_mask,
            update_queue=False, force_image_path=None,
        )
        loss = criterion(logits_fuse, labels)
        total_loss += loss.item()
        preds_fuse.extend(torch.argmax(logits_fuse, dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        # B) 全部强制图像头（专门监控 img_head）
        force = torch.ones(b, device=device)
        logits_img = model(
            images, input_ids, attention_mask,
            update_queue=False, force_image_path=force,
        )
        preds_img.extend(torch.argmax(logits_img, dim=1).cpu().numpy())

    acc_fuse = accuracy_score(labels_all, preds_fuse)
    f1_fuse = f1_score(labels_all, preds_fuse, average="weighted")
    acc_img = accuracy_score(labels_all, preds_img)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc_fuse, f1_fuse, acc_img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Model: LAD-DRF | 改法1+2")
    print(f"FORCE_IMAGE_PROB={FORCE_IMAGE_PROB}  TEXT_GATE_THRESHOLD={TEXT_GATE_THRESHOLD}")
    print(f"保存: {SAVE_PATH}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader = DataLoader(
        MVSADataset(train_json, image_dir, tokenizer, is_train=True),
        batch_size=16, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        MVSADataset(val_json, image_dir, tokenizer, is_train=False),
        batch_size=16, shuffle=False, num_workers=0,
    )

    model = LADDRF(
        num_classes=3,
        text_gate_threshold=TEXT_GATE_THRESHOLD,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    best_score = -1.0

    for epoch in range(6):
        print(f"\n===== Epoch {epoch+1}/6 =====")
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        va_loss, va_acc, va_f1, va_acc_img = evaluate(
            model, val_loader, criterion, device
        )

        # 改法2：融合 Val 与强制图像 Val 一起选点
        #score = 0.5 * va_acc + 0.5 * va_acc_img
        score = 0.6 * va_acc + 0.4 * va_acc_img

        print(f"FORCE={FORCE_IMAGE_PROB}  SCORE={score}")
        print(f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} F1={tr_f1:.4f}")
        print(
            f"Val   Loss={va_loss:.4f} Acc_fuse={va_acc:.4f} F1={va_f1:.4f} "
            f"| Acc_img={va_acc_img:.4f} | score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), SAVE_PATH)
            print(f">> 保存 {SAVE_PATH} (score={score:.4f})")

    print(f"\n完成。最佳 score={best_score:.4f}")
    print(f"权重: {SAVE_PATH}")


if __name__ == "__main__":
    main()