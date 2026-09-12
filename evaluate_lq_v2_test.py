import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from dataset import MVSADataset
from models.lq_drf_v2 import LQDRFv2

data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
test_json = os.path.join(data_root, "test.json")
model_path = "best_lq_drf_v2.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    loader = DataLoader(
        MVSADataset(test_json, image_dir, tokenizer, is_train=False),
        batch_size=16, shuffle=False, num_workers=0
    )
    model = LQDRFv2(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            logits = model(
                batch["image"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                update_queue=False,
            )
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch["label"].numpy())

    print(f"Acc: {accuracy_score(labels, preds):.4f}")
    print(f"W-F1: {f1_score(labels, preds, average='weighted'):.4f}")
    print(f"Macro-F1: {f1_score(labels, preds, average='macro'):.4f}")
    print(classification_report(labels, preds, target_names=["neg", "neu", "pos"], digits=4))

if __name__ == "__main__":
    main()