import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer


class MVSADataset(Dataset):
    def __init__(self, json_path, image_dir, tokenizer, max_length=64, is_train=True):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 图像
        img_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 文本
        text = item["text"]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        label = item["label"]

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }