import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from dataset import MVSADataset
from models.lightweight_drf import LightweightDRF

# ==================== 路径配置 ====================
data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
test_json = os.path.join(data_root, "test.json")
model_path = "best_lightweight_drf.pth"


# ================================================

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, input_ids, attention_mask, update_queue=False)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print("\n===== 测试集最终结果（Lightweight DRF） =====")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Weighted-F1  : {weighted_f1:.4f}")
    print(f"Macro-F1     : {macro_f1:.4f}")
    print("\n分类报告：")
    print(classification_report(all_labels, all_preds,
                                target_names=["negative", "neutral", "positive"],
                                digits=4))

    return acc, weighted_f1, macro_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    test_dataset = MVSADataset(test_json, image_dir, tokenizer, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = LightweightDRF(num_classes=3, queue_size=512).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    else:
        print(f"找不到模型文件: {model_path}")
        return

    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()