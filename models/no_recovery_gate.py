import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from transformers import DistilBertModel
from collections import deque


class FeatureQueue:
    def __init__(self, max_size=512, feature_dim=512):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)

    def update(self, features):
        for f in features.detach().cpu():
            self.queue.append(f)

    def get_mean_std(self):
        if len(self.queue) < 10:
            return None, None
        feats = torch.stack(list(self.queue))
        mean = feats.mean(dim=0)
        std = feats.std(dim=0) + 1e-6
        return mean, std


class NoRecoveryGate(nn.Module):
    """
    质量加权 + 文本门控：
    文本质量低于阈值时，降低或关闭文本权重，更多依赖图像。
    """
    def __init__(
        self,
        num_classes=3,
        queue_size=512,
        text_model_name="distilbert-base-uncased",
        text_gate_threshold=0.35,
        hard_gate=True,  # True: 低于阈值直接关掉文本; False: 软门控
    ):
        super().__init__()
        self.text_gate_threshold = text_gate_threshold
        self.hard_gate = hard_gate

        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.image_encoder = mobilenet_v2(weights=weights)
        self.image_encoder.classifier = nn.Identity()
        self.image_dim = 1280

        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        self.text_dim = 768

        self.img_proj = nn.Linear(self.image_dim, 512)
        self.txt_proj = nn.Linear(self.text_dim, 512)

        self.img_queue = FeatureQueue(max_size=queue_size, feature_dim=512)
        self.txt_queue = FeatureQueue(max_size=queue_size, feature_dim=512)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes),
        )

    def compute_quality(self, feat, mean, std):
        if mean is None or std is None:
            return torch.ones(feat.size(0), device=feat.device)
        dist = torch.sqrt(
            ((feat - mean.to(feat.device)) / std.to(feat.device)).pow(2).sum(dim=1)
        )
        return torch.exp(-dist / 2.0)

    def forward(self, image, input_ids, attention_mask, update_queue=True):
        img_feat = self.image_encoder(image)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state[:, 0, :]

        img_feat = self.img_proj(img_feat)
        txt_feat = self.txt_proj(txt_feat)

        if update_queue and self.training:
            self.img_queue.update(img_feat)
            self.txt_queue.update(txt_feat)

        img_mean, img_std = self.img_queue.get_mean_std()
        txt_mean, txt_std = self.txt_queue.get_mean_std()

        img_q = self.compute_quality(img_feat, img_mean, img_std)
        txt_q = self.compute_quality(txt_feat, txt_mean, txt_std)
        img_q = 0.6 * img_q + 0.4
        txt_q = 0.6 * txt_q + 0.4

        # 基础质量加权
        qs = img_q + txt_q + 1e-6
        img_w = img_q / qs
        txt_w = txt_q / qs

        # ----- 文本门控 -----
        if self.hard_gate:
            # 质量低于阈值：文本权重置 0，全给图像
            gate = (txt_q >= self.text_gate_threshold).float()
            txt_w = txt_w * gate
            img_w = 1.0 - txt_w
        else:
            # 软门控：再乘一次文本质量
            txt_w = txt_w * txt_q
            img_w = img_w * img_q
            s = img_w + txt_w + 1e-6
            img_w, txt_w = img_w / s, txt_w / s

        img_w = img_w.unsqueeze(1)
        txt_w = txt_w.unsqueeze(1)

        fused = torch.cat([img_feat * img_w, txt_feat * txt_w], dim=1)
        return self.classifier(fused)