import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import MVSADataset
from models.lightweight_drf_ablation import LightweightDRFAblation

# ==================== 路径 ====================
data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
test_json = os.path.join(data_root, "test.json")
# ==============================================

ABLATIONS = {
    "full":        (True,  True,  "best_ablation_full.pth"),
    "no_recovery": (True,  False, "best_ablation_no_recovery.pth"),
    "no_quality":  (False, True,  "best_ablation_no_quality.pth"),
    "no_both":     (False, False, "best_ablation_no_both.pth"),
}

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Testing", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(images, input_ids, attention_mask, update_queue=False)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    w_f1 = f1_score(all_labels, all_preds, average="weighted")
    m_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, w_f1, m_f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    test_dataset = MVSADataset(test_json, image_dir, tokenizer, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"{'版本':<15} {'Acc':>8} {'W-F1':>8} {'Macro-F1':>10}")
    print("-" * 45)

    for name, (use_qw, use_rec, ckpt) in ABLATIONS.items():
        if not os.path.exists(ckpt):
            print(f"{name:<15} 文件不存在: {ckpt}")
            continue

        model = LightweightDRFAblation(
            num_classes=3,
            use_quality_weight=use_qw,
            use_recovery=use_rec,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        acc, w_f1, m_f1 = evaluate(model, test_loader, device)
        print(f"{name:<15} {acc:8.4f} {w_f1:8.4f} {m_f1:10.4f}")

    print("\n消融测试集评估完成。")

if __name__ == "__main__":
    main()