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

FORCE_IMAGE_PROB = 0.35
MASK_RATIO_THR = 0.5
SAVE_PATH = "best_lad_drf_v1.pth"  # 另存，避免再覆盖


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        bsz = labels.size(0)
        # 旧配方：只随机强制走图像头，不改 input_ids
        force = (torch.rand(bsz, device=device) < FORCE_IMAGE_PROB)

        optimizer.zero_grad()
        logits = model(
            images,
            input_ids,
            attention_mask,
            update_queue=True,
            force_image_path=force,
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(
            images,
            input_ids,
            attention_mask,
            update_queue=False,
            force_image_path=None,  # 验证用规则路由
        )
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("训练: LAD-DRF 旧配方（只 force 图像路径，不改 input 为 MASK）")
    print(f"FORCE_IMAGE_PROB={FORCE_IMAGE_PROB}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = MVSADataset(train_json, image_dir, tokenizer, is_train=True)
    val_dataset = MVSADataset(val_json, image_dir, tokenizer, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 1️⃣ 模型初始化时传入 tokenizer（如果 LADDRF.__init__ 需要它）
    model = LADDRF(
        num_classes=3,
        mask_ratio_thr=MASK_RATIO_THR,
        tokenizer=tokenizer,   # 关键！传入 tokenizer
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    save_path = "best_lad_drf_v1.pth"
    best_val_acc = 0.0

    for epoch in range(6):
        print(f"\n===== Epoch {epoch+1}/6 =====")
        # 2️⃣ 调用时去掉 tokenizer（函数不需要它）
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f">> 保存最佳模型: {save_path}")

    print(f"\n训练完成！最佳 Val Acc: {best_val_acc:.4f}")
    print(f"权重文件: {save_path}")


if __name__ == "__main__":
    main()