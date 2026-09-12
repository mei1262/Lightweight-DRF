import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import MVSADataset
from models.lq_drf_rec import LQDRFRec

# ==================== 路径 ====================
data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
train_json = os.path.join(data_root, "train.json")
val_json = os.path.join(data_root, "val.json")
# ==============================================

# 损失权重（可调）
REC_WEIGHT = 0.3
CON_WEIGHT = 0.1
USE_CONTRASTIVE = True   # 只想先试重建，改成 False


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_rec, total_con = 0, 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, rec_loss, con_loss = model(
            images, input_ids, attention_mask, update_queue=True
        )
        cls_loss = criterion(logits, labels)
        loss = cls_loss + REC_WEIGHT * rec_loss + CON_WEIGHT * con_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rec += float(rec_loss.detach().cpu())
        total_con += float(con_loss.detach().cpu())
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / n, total_rec / n, total_con / n, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
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
    print("Model: LQ-DRF-Rec (重建损失" + (" + 对比学习" if USE_CONTRASTIVE else "") + ")")
    print(f"REC_WEIGHT={REC_WEIGHT} CON_WEIGHT={CON_WEIGHT}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    # 用干净 Dataset；破坏在模型特征空间内完成
    train_dataset = MVSADataset(train_json, image_dir, tokenizer, is_train=True)
    val_dataset = MVSADataset(val_json, image_dir, tokenizer, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = LQDRFRec(
        num_classes=3,
        use_contrastive=USE_CONTRASTIVE,
        disrupt_prob=0.5,
        rec_loss_weight=REC_WEIGHT,
        con_loss_weight=CON_WEIGHT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    save_path = "best_lq_drf_rec.pth"

    for epoch in range(6):
        print(f"\n===== Epoch {epoch+1}/6 =====")
        tr_loss, tr_rec, tr_con, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, device)

        print(
            f"Train Loss={tr_loss:.4f} Rec={tr_rec:.4f} Con={tr_con:.4f} "
            f"| Acc={tr_acc:.4f} F1={tr_f1:.4f}"
        )
        print(f"Val   Loss={va_loss:.4f} | Acc={va_acc:.4f} F1={va_f1:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), save_path)
            print(f">> 保存 {save_path}")

    print(f"\n完成。最佳 Val Acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()