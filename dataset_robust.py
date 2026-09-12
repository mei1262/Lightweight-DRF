import os
import json
import random
from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer

from disruptions import (
    DisruptionStrategy,
    DisruptionType,
    corrupt_image,
    corrupt_text,
    decide_disruption,
    make_missing_image,
    make_missing_text,
    sample_corruption_ratio,
)


class RobustMVSADataset(Dataset):
    """
    MVSA dataset with on-the-fly low-quality / missing-modality disruptions.

    Follows the DRF paper protocol:
    - fixed: disrupt one modality for all samples (inference-time robustness)
    - random: disrupt a random modality for ``disruption_ratio`` of samples
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        tokenizer: DistilBertTokenizer,
        max_length: int = 64,
        is_train: bool = False,
        strategy: Optional[DisruptionStrategy] = None,
        disruption_type: DisruptionType = DisruptionType.C,
        disruption_ratio: float = 1.0,
        disrupted_modality: str = "image",
        corrupt_ratio_range: Tuple[float, float] = (0.4, 0.8),
        seed: int = 42,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.strategy = strategy
        self.disruption_type = disruption_type
        self.disruption_ratio = disruption_ratio
        self.disrupted_modality = disrupted_modality
        self.corrupt_ratio_range = corrupt_ratio_range
        self.seed = seed
        self._rng = random.Random(seed)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if is_train and strategy == DisruptionStrategy.RANDOM:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def _load_sample(self, idx: int):
        item = self.data[idx]

        img_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        encoding = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = item["label"]

        meta = {
            "id": item.get("id", str(idx)),
            "disrupted": False,
            "disrupted_modality": None,
            "operation": None,
        }
        return image, input_ids, attention_mask, label, meta

    def _apply_disruption(
        self,
        image,
        input_ids,
        attention_mask,
        idx: int,
    ):
        if self.strategy is None:
            return image, input_ids, attention_mask, {
                "disrupted": False,
                "disrupted_modality": None,
                "operation": None,
            }

        sample_rng = random.Random(self.seed + idx)
        is_disrupted, target_modality, operation = decide_disruption(
            sample_index=idx,
            strategy=self.strategy,
            disruption_type=self.disruption_type,
            disruption_ratio=self.disruption_ratio,
            disrupted_modality=self.disrupted_modality,
            rng=sample_rng,
        )

        meta = {
            "disrupted": is_disrupted,
            "disrupted_modality": target_modality,
            "operation": operation,
        }

        if not is_disrupted:
            return image, input_ids, attention_mask, meta

        if operation == "discard":
            if target_modality == "image":
                image = make_missing_image(image)
            else:
                input_ids, attention_mask = make_missing_text(
                    input_ids, attention_mask, self.mask_token_id
                )
            return image, input_ids, attention_mask, meta

        corruption_ratio = sample_corruption_ratio(
            self.corrupt_ratio_range, rng=sample_rng
        )
        if target_modality == "image":
            image = corrupt_image(image, corruption_ratio, rng=sample_rng)
        else:
            input_ids, attention_mask = corrupt_text(
                input_ids,
                attention_mask,
                self.mask_token_id,
                corruption_ratio,
                self.pad_token_id,
                rng=sample_rng,
            )

        return image, input_ids, attention_mask, meta

    def __getitem__(self, idx):
        image, input_ids, attention_mask, label, base_meta = self._load_sample(idx)
        image, input_ids, attention_mask, disruption_meta = self._apply_disruption(
            image, input_ids, attention_mask, idx
        )

        meta = {**base_meta, **disruption_meta}
        # 把 None 转成字符串，避免 collate 报错
        safe_meta = {
            "id": str(meta.get("id", "")),
            "disrupted": bool(meta.get("disrupted", False)),
            "disrupted_modality": str(meta.get("disrupted_modality") or "none"),
            "operation": str(meta.get("operation") or "none"),
        }

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "meta": safe_meta,
        }
