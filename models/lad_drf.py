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


class LADDRF(nn.Module):
    """
    Lightweight Asymmetric DRF (路线2：双路径路由)
    - 文本质量高：质量加权融合头（接近 no_recovery）
    - 文本质量低：纯图像分类头
    训练时可强制走图像路径（模拟文本失效）
    """
    def __init__(
        self,
        num_classes=3,
        queue_size=512,
        text_model_name="distilbert-base-uncased",
        text_gate_threshold=0.35,
    ):
        super().__init__()
        self.text_gate_threshold = text_gate_threshold

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

        # 路径1：质量加权融合
        self.fuse_head = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes),
        )
        # 路径2：纯图像
        self.img_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def compute_quality(self, feat, mean, std):
        if mean is None or std is None:
            return torch.ones(feat.size(0), device=feat.device)
        dist = torch.sqrt(
            ((feat - mean.to(feat.device)) / std.to(feat.device)).pow(2).sum(dim=1)
        )
        return torch.exp(-dist / 2.0)

    def encode(self, image, input_ids, attention_mask, update_queue=True):
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
        return img_feat, txt_feat, img_q, txt_q

    def fuse_logits(self, img_feat, txt_feat, img_q, txt_q):
        qs = img_q + txt_q + 1e-6
        img_w = (img_q / qs).unsqueeze(1)
        txt_w = (txt_q / qs).unsqueeze(1)
        fused = torch.cat([img_feat * img_w, txt_feat * txt_w], dim=1)
        return self.fuse_head(fused)

    def forward(
        self,
        image,
        input_ids,
        attention_mask,
        update_queue=True,
        force_image_path=None,
    ):
        """
        force_image_path:
          - None: 按文本质量自动路由（推理默认）
          - Bool tensor [B]: True 走图像头（训练时模拟文本失效）
        """
        img_feat, txt_feat, img_q, txt_q = self.encode(
            image, input_ids, attention_mask, update_queue=update_queue
        )

        logits_fuse = self.fuse_logits(img_feat, txt_feat, img_q, txt_q)
        logits_img = self.img_head(img_feat)

        if force_image_path is None:
            # 推理：文本质量低 → 图像头
            use_img = txt_q < self.text_gate_threshold  # [B]
        else:
            use_img = force_image_path.bool()

        use_img = use_img.view(-1, 1).float()
        logits = use_img * logits_img + (1.0 - use_img) * logits_fuse
        return logits