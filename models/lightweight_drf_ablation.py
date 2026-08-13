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


class LightweightDRFAblation(nn.Module):
    """
    专门用于消融实验的模型。
    use_quality_weight: 是否使用质量加权
    use_recovery: 是否使用残差恢复
    """
    def __init__(
        self,
        num_classes=3,
        queue_size=512,
        text_model_name="distilbert-base-uncased",
        use_quality_weight=True,
        use_recovery=True,
    ):
        super().__init__()
        self.use_quality_weight = use_quality_weight
        self.use_recovery = use_recovery

        # 图像编码器
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.image_encoder = mobilenet_v2(weights=weights)
        self.image_encoder.classifier = nn.Identity()
        self.image_dim = 1280

        # 文本编码器
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        self.text_dim = 768

        # 投影
        self.img_proj = nn.Linear(self.image_dim, 512)
        self.txt_proj = nn.Linear(self.text_dim, 512)

        # 特征队列
        self.img_queue = FeatureQueue(max_size=queue_size, feature_dim=512)
        self.txt_queue = FeatureQueue(max_size=queue_size, feature_dim=512)

        # 恢复网络
        self.img_to_txt = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.txt_to_img = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.recovery_alpha = 0.5

        # 分类头
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
        quality = torch.exp(-dist / 2.0)
        return quality

    def forward(self, image, input_ids, attention_mask, update_queue=True):
        # 1. 提取特征
        img_feat = self.image_encoder(image)
        text_output = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        txt_feat = text_output.last_hidden_state[:, 0, :]

        # 2. 投影
        img_feat = self.img_proj(img_feat)
        txt_feat = self.txt_proj(txt_feat)

        # 3. 更新队列
        if update_queue and self.training:
            self.img_queue.update(img_feat)
            self.txt_queue.update(txt_feat)

        # 4. 计算质量
        img_mean, img_std = self.img_queue.get_mean_std()
        txt_mean, txt_std = self.txt_queue.get_mean_std()

        img_quality = self.compute_quality(img_feat, img_mean, img_std)
        txt_quality = self.compute_quality(txt_feat, txt_mean, txt_std)
        img_quality = 0.5 * img_quality + 0.5
        txt_quality = 0.5 * txt_quality + 0.5

        # 5. 残差恢复（可消融）
        if self.use_recovery:
            recovered_img = self.txt_to_img(txt_feat)
            recovered_txt = self.img_to_txt(img_feat)
            img_feat = img_feat + self.recovery_alpha * recovered_img * (
                1.0 - img_quality
            ).unsqueeze(1)
            txt_feat = txt_feat + (self.recovery_alpha * 1.5) * recovered_txt * (
                1.0 - txt_quality
            ).unsqueeze(1)

        # 6. 融合（可消融质量加权）
        if self.use_quality_weight:
            quality_sum = img_quality + txt_quality + 1e-6
            img_weight = (img_quality / quality_sum).unsqueeze(1)
            txt_weight = (txt_quality / quality_sum).unsqueeze(1)
        else:
            img_weight = torch.ones(img_feat.size(0), 1, device=img_feat.device) * 0.5
            txt_weight = torch.ones(txt_feat.size(0), 1, device=txt_feat.device) * 0.5

        fused = torch.cat([img_feat * img_weight, txt_feat * txt_weight], dim=1)
        logits = self.classifier(fused)
        return logits