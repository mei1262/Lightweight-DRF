import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import MVSADataset
from models.lightweight_drf_ablation import LightweightDRFAblation

data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
train_json = os.path.join(data_root, "train.json")
val_json = os.path.join(data_root, "val.json")

# 选一种损失: "weighted" 或 "focal"
LOSS_TYPE = "focal"  # 先试 weighted；再改 focal

# 类别权重: [negative, neutral, positive] —— neutral 通常最少，权重大一些
CLASS_WEIGHTS = [1.2, 2.5, 1.0]

# Focal 参数
FOCAL_GAMMA = 2.0


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds, labels = 0, [], []
    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images, input_ids, attention_mask, update_queue=True)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    return total_loss / len(loader), accuracy_score(labels, preds), f1_score(labels, preds, average="weighted")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0, [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        logits = model(images, input_ids, attention_mask, update_queue=False)
        loss = criterion(logits, y)
        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    acc = accuracy_score(labels, preds)
    w_f1 = f1_score(labels, preds, average="weighted")
    m_f1 = f1_score(labels, preds, average="macro")
    return total_loss / len(loader), acc, w_f1, m_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"LOSS_TYPE={LOSS_TYPE}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader = DataLoader(
        MVSADataset(train_json, image_dir, tokenizer, is_train=True),
        batch_size=16, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        MVSADataset(val_json, image_dir, tokenizer, is_train=False),
        batch_size=16, shuffle=False, num_workers=0,
    )

    model = LightweightDRFAblation(
        num_classes=3,
        use_quality_weight=True,
        use_recovery=False,
    ).to(device)

    w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    if LOSS_TYPE == "focal":
        criterion = FocalLoss(weight=w, gamma=FOCAL_GAMMA)
        save_path = "best_no_recovery_focal.pth"
    else:
        criterion = nn.CrossEntropyLoss(weight=w)
        save_path = "best_no_recovery_weighted.pth"

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.05)
    best = 0.0

    for epoch in range(6):
        print(f"\n===== Epoch {epoch+1}/6 =====")
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_wf1, va_mf1 = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} F1={tr_f1:.4f}")
        print(f"Val   Loss={va_loss:.4f} Acc={va_acc:.4f} W-F1={va_wf1:.4f} Macro-F1={va_mf1:.4f}")
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), save_path)
            print(f">> 保存 {save_path}")

    print(f"\n完成。最佳 Val Acc={best:.4f} | 权重={save_path}")


if __name__ == "__main__":
    main()