# Lightweight-DRF

**Lightweight Robust Multimodal Sentiment Analysis for Image-Text Pairs via Simplified Distribution-Based Feature Recovery and Fusion**

基于 [DRF (ACM MM 2024)](https://doi.org/10.1145/3664647.3680653) 的轻量化鲁棒图文情感分析实现，在 MVSA-Single 数据集上验证低质量与缺失模态场景下的鲁棒性。

---

## 最终模型：LAD-DRF

> **本仓库的最终模型是 LAD-DRF（Lightweight Asymmetric DRF），而非 Baseline 或早期 Lightweight DRF 变体。**
>
> 论文实验、性能对比与复现请以 **LAD-DRF** 为准。其余模型文件仅作为对照或消融探索保留。

| 角色 | 文件 | 说明 |
|------|------|------|
| **最终模型（主方法）** | `models/lad_drf.py` | 双路径不对称路由：质量加权融合头 + 纯图像头 |
| **训练** | `train_lad_drf.py` | 训练 LAD-DRF，保存 `best_lad_drf_*.pth` |
| **日常评估** | `evaluate_lad_drf_rule_route.py` | 规则路由，快速查看 clean / fixed / Text-D |
| **论文评估** | `evaluate_lad_drf_official.py` | 质量自动路由 + Random 全 sweep，输出 CSV |

### LAD-DRF 核心思路

- **骨干**：MobileNetV2 + DistilBERT（轻量）
- **分布队列**：近似各模态特征分布，用于质量估计
- **双路径路由**：
  - 文本质量高 → 质量加权融合头
  - 文本质量低 / 缺失 → 纯图像分类头
- **训练**：以一定概率强制走图像路径，模拟文本失效

### 对照 / 历史模型（非最终方案）

| 模型 | 文件 | 用途 |
|------|------|------|
| Baseline | `models/baseline.py`, `train.py` | 简单拼接融合基线 |
| Lightweight DRF | `models/lightweight_drf.py`, `train_drf.py` | 早期简化 DRF（质量加权弱化版） |
| 消融变体 | `models/lightweight_drf_ablation.py` 等 | 组件消融实验 |
| 其他探索 | `models/lq_drf_*.py`, `no_recovery_gate.py` 等 | 中间迭代，仅供参考 |

---

## 环境

```bash
pip install -r requirements.txt
```

需要 Python 3.8+，建议使用 CUDA 环境。

---

## 数据准备

1. 下载 [MVSA-Single](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) 原始数据
2. 将图片与 `labelResultAll.txt` 放到 `data/MVSA_Single/` 下
3. 运行预处理：

```bash
python preprocess_mvsa.py
```

生成 `train.json` / `val.json` / `test.json`（已随仓库提供划分文件；原始图片因体积未纳入 git）。

---

## 训练

```bash
# Baseline（对照）
python train.py

# LAD-DRF（最终模型）
python train_lad_drf.py
```

训练参数（`train_lad_drf.py` 顶部）：

- `FORCE_IMAGE_PROB`：训练时强制图像路径的概率
- `TEXT_GATE_THRESHOLD`：推理时文本质量门控阈值
- `SAVE_PATH`：checkpoint 保存路径

---

## 评估

### 鲁棒性协议（对齐 DRF 论文）

- **Fixed**：测试时固定干扰某一模态（C=低质量，D=缺失，C+D=混合）
- **Random**：按 disruption ratio（0.2–1.0）随机干扰

```bash
# LAD-DRF 日常评估（规则路由）
python evaluate_lad_drf_rule_route.py

# LAD-DRF 论文级评估（质量路由 + Random sweep）
python evaluate_lad_drf_official.py

# Baseline / 早期 DRF 鲁棒性对比
python evaluate_robust.py --mode all --full-sweep --models baseline drf
```

结果保存在 `results/` 目录。

---

## 项目结构

```
Lightweight-DRF/
├── models/
│   ├── lad_drf.py              # ★ 最终模型
│   ├── baseline.py             # 基线
│   └── lightweight_drf.py      # 早期变体
├── train_lad_drf.py            # ★ LAD-DRF 训练
├── evaluate_lad_drf_rule_route.py  # ★ 日常评估
├── evaluate_lad_drf_official.py    # ★ 论文评估
├── dataset.py / dataset_robust.py
├── disruptions.py              # 干扰策略（C / D / C+D, fixed / random）
├── preprocess_mvsa.py
└── results/                    # 实验结果 CSV
```

---

## 引用

若使用本仓库，请引用原始 DRF 论文：

```bibtex
@inproceedings{drf2024mm,
  title={Robust Multimodal Sentiment Analysis of Image-Text Pairs by Distribution-Based Feature Recovery and Fusion},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)},
  year={2024},
  doi={10.1145/3664647.3680653}
}
```

---

## License

Research use only. MVSA 数据集请遵循其原始许可协议。
