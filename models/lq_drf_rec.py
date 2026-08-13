import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LQDRFRec(nn.Module):
    """
    LQ-DRF-Rec: 质量加权 + 恢复 + 重建损失（可选对比学习）
    训练时在特征空间随机破坏某一模态，用另一模态恢复，并与完好特征做 MSE。
    """
    def __init__(
        self,
        num_classes=3,
        queue_size=512,
        text_model_name="distilbert-base-uncased",
        use_contrastive=True,
        disrupt_prob=0.5,
        rec_loss_weight=0.3,
        con_loss_weight=0.1,
        temperature=0.07,
    ):
        super().__init__()
        self.use_contrastive = use_contrastive
        self.disrupt_prob = disrupt_prob
        self.rec_loss_weight = rec_loss_weight
        self.con_loss_weight = con_loss_weight
        self.temperature = temperature

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

        # 恢复网络
        self.img_to_txt = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)
        )
        self.txt_to_img = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)
        )

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

    def contrastive_loss(self, img_feat, txt_feat):
        """batch 内图文 InfoNCE"""
        img = F.normalize(img_feat, dim=-1)
        txt = F.normalize(txt_feat, dim=-1)
        logits = img @ txt.t() / self.temperature
        labels = torch.arange(img.size(0), device=img.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i2t + loss_t2i)

    def forward(self, image, input_ids, attention_mask, update_queue=True):
        # 1) 编码
        img_feat = self.image_encoder(image)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state[:, 0, :]

        img_feat = self.img_proj(img_feat)
        txt_feat = self.txt_proj(txt_feat)

        # 完好特征（重建目标，不反传到目标侧）
        img_clean = img_feat
        txt_clean = txt_feat

        rec_loss = img_feat.new_tensor(0.0)
        con_loss = img_feat.new_tensor(0.0)

        # 2) 训练期：特征级随机破坏 + 重建
        if self.training and self.disrupt_prob > 0:
            b = img_feat.size(0)
            device = img_feat.device
            # 每个样本是否破坏
            do_disrupt = torch.rand(b, device=device) < self.disrupt_prob
            # 破坏图像(0)还是文本(1)
            disrupt_txt = torch.rand(b, device=device) < 0.5

            img_bad = img_feat.clone()
            txt_bad = txt_feat.clone()

            # 破坏：置零（模拟缺失）；也可用噪声，这里用缺失更贴近 Fixed-D
            img_mask = do_disrupt & (~disrupt_txt)
            txt_mask = do_disrupt & disrupt_txt
            img_bad[img_mask] = 0.0
            txt_bad[txt_mask] = 0.0

            # 恢复
            img_rec = self.txt_to_img(txt_bad)
            txt_rec = self.img_to_txt(img_bad)

            # 只在被破坏位置算重建损失
            if img_mask.any():
                rec_loss = rec_loss + F.mse_loss(img_rec[img_mask], img_clean[img_mask].detach())
            if txt_mask.any():
                rec_loss = rec_loss + F.mse_loss(txt_rec[txt_mask], txt_clean[txt_mask].detach())

            # 用恢复结果替换被破坏特征
            img_feat = torch.where(img_mask.unsqueeze(1), img_rec, img_feat)
            txt_feat = torch.where(txt_mask.unsqueeze(1), txt_rec, txt_feat)
        else:
            # 推理：质量低时轻量残差恢复
            img_mean, img_std = self.img_queue.get_mean_std()
            txt_mean, txt_std = self.txt_queue.get_mean_std()
            img_q = self.compute_quality(img_feat, img_mean, img_std)
            txt_q = self.compute_quality(txt_feat, txt_mean, txt_std)
            img_q = 0.6 * img_q + 0.4
            txt_q = 0.6 * txt_q + 0.4

            img_rec = self.txt_to_img(txt_feat)
            txt_rec = self.img_to_txt(img_feat)
            img_low = (img_q < 0.35).float().unsqueeze(1)
            txt_low = (txt_q < 0.35).float().unsqueeze(1)
            img_feat = img_feat + img_low * 0.3 * img_rec * (1.0 - img_q).unsqueeze(1)
            txt_feat = txt_feat + txt_low * 0.1 * txt_rec * (1.0 - txt_q).unsqueeze(1)

        # 3) 更新队列（用当前特征）
        if update_queue and self.training:
            self.img_queue.update(img_feat)
            self.txt_queue.update(txt_feat)

        # 4) 质量加权融合
        img_mean, img_std = self.img_queue.get_mean_std()
        txt_mean, txt_std = self.txt_queue.get_mean_std()
        img_q = self.compute_quality(img_feat, img_mean, img_std)
        txt_q = self.compute_quality(txt_feat, txt_mean, txt_std)
        img_q = 0.6 * img_q + 0.4
        txt_q = 0.6 * txt_q + 0.4

        qs = img_q + txt_q + 1e-6
        img_w = (img_q / qs).unsqueeze(1)
        txt_w = (txt_q / qs).unsqueeze(1)

        fused = torch.cat([img_feat * img_w, txt_feat * txt_w], dim=1)
        logits = self.classifier(fused)

        # 5) 可选对比学习（在完好投影特征上）
        if self.training and self.use_contrastive:
            con_loss = self.contrastive_loss(img_clean, txt_clean)

        if self.training:
            return logits, rec_loss, con_loss
        return logits