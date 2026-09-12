import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import MVSADataset
from dataset_robust import RobustMVSADataset
from disruptions import DisruptionStrategy, DisruptionType
from models.lightweight_drf_ablation import LightweightDRFAblation

# ==================== 路径 ====================
data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
train_json = os.path.join(data_root, "train.json")
val_json = os.path.join(data_root, "val.json")
# ==============================================

# 训练期干扰强度：建议先 0.4，太大容易伤干净精度
TRAIN_DISRUPTION_RATIO = 0.4


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images, input_ids, attention_mask, update_queue=True)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(images, input_ids, attention_mask, update_queue=False)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("模型: no_recovery | 训练: Random 干扰增强")
    print(f"TRAIN_DISRUPTION_RATIO = {TRAIN_DISRUPTION_RATIO}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # ----- 训练集：在线 Random 干扰 -----
    train_dataset = RobustMVSADataset(
        json_path=train_json,
        image_dir=image_dir,
        tokenizer=tokenizer,
        is_train=True,
        strategy=DisruptionStrategy.RANDOM,
        disruption_type=DisruptionType.C_D,  # 损坏和缺失都有
        disruption_ratio=TRAIN_DISRUPTION_RATIO,
        disrupted_modality="image",  # random 模式下会随机选模态
        seed=42,
    )

    # ----- 验证集：保持干净，监控有没有伤太重 -----
    val_dataset = MVSADataset(val_json, image_dir, tokenizer, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = LightweightDRFAblation(
        num_classes=3,
        use_quality_weight=True,
        use_recovery=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    save_path = "best_no_recovery_aug.pth"  # 新文件，不覆盖旧模型

    for epoch in range(6):
        print(f"\n===== Epoch {epoch+1}/6 =====")
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f">> 保存: {save_path}")

    print(f"\n完成。最佳 Val Acc: {best_val_acc:.4f}")
    print(f"权重文件: {save_path}")


if __name__ == "__main__":
    main()