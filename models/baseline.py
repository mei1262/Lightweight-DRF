import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from transformers import DistilBertModel


class BaselineModel(nn.Module):
    def __init__(self, num_classes=3, text_model_name="distilbert-base-uncased"):
        super().__init__()

        # 图像编码器
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.image_encoder = mobilenet_v2(weights=weights)
        self.image_encoder.classifier = nn.Identity()  # 去掉原来的分类头
        self.image_dim = 1280

        # 文本编码器
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        self.text_dim = 768

        # 融合 + 分类
        self.classifier = nn.Sequential(
            nn.Linear(self.image_dim + self.text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # 图像特征
        img_feat = self.image_encoder(image)  # [B, 1280]

        # 文本特征
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_output.last_hidden_state[:, 0, :]  # [CLS] token  [B, 768]

        # 拼接融合
        fused = torch.cat([img_feat, text_feat], dim=1)

        logits = self.classifier(fused)
        return logits