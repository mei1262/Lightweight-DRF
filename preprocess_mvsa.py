import os
import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ==================== 需要修改的路径 ====================
# 原始数据所在文件夹（里面有 1.jpg, 1.txt, 2.jpg...）
raw_data_dir = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single\data"  # ← 改成你的实际路径

# 标注文件路径
label_file = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single\labelResultAll.txt"  # ← 改成你的实际路径

# 输出目录
output_dir = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
# ======================================================

os.makedirs(output_dir, exist_ok=True)

# 标签映射
label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


def get_final_label(text_label, image_label):
    """
    按照通用规则确定最终标签：
    1. 如果一个 positive、一个 negative → 冲突，返回 None（丢弃）
    2. 如果有一个是 neutral，取另一个非中性标签
    3. 两个相同，直接取
    """
    if text_label == image_label:
        return text_label

    # 冲突：一正一负
    if (text_label == "positive" and image_label == "negative") or \
            (text_label == "negative" and image_label == "positive"):
        return None

    # 有一个是 neutral，取另一个
    if text_label == "neutral":
        return image_label
    if image_label == "neutral":
        return text_label

    return None


# 1. 读取标注文件
print("正在读取标注文件...")
samples = []
with open(label_file, "r", encoding="utf-8") as f:
    lines = f.readlines()[1:]  # 跳过第一行表头

for line in lines:
    line = line.strip()
    if not line:
        continue
    parts = line.split("\t")
    if len(parts) != 2:
        continue

    img_id = parts[0].strip()
    labels = parts[1].strip().split(",")
    if len(labels) != 2:
        continue

    text_label = labels[0].strip().lower()
    image_label = labels[1].strip().lower()

    final_label = get_final_label(text_label, image_label)
    if final_label is None:
        continue  # 冲突样本直接丢弃

    # 检查对应的图片和文本是否存在
    img_path = os.path.join(raw_data_dir, f"{img_id}.jpg")
    txt_path = os.path.join(raw_data_dir, f"{img_id}.txt")

    if not (os.path.exists(img_path) and os.path.exists(txt_path)):
        continue

    # 读取文本内容
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as tf:
        text = tf.read().strip()

    samples.append({
        "id": img_id,
        "image": f"{img_id}.jpg",
        "text": text,
        "label": label2id[final_label]
    })

print(f"有效样本数: {len(samples)}")

# 2. 按 8:1:1 划分（固定随机种子，保证可复现）
random.seed(42)
train_val, test = train_test_split(samples, test_size=0.1, random_state=42, stratify=[s["label"] for s in samples])
train, val = train_test_split(train_val, test_size=1 / 9, random_state=42,
                              stratify=[s["label"] for s in train_val])  # 1/9 ≈ 使整体接近 8:1:1

print(f"训练集: {len(train)}")
print(f"验证集: {len(val)}")
print(f"测试集: {len(test)}")


# 3. 保存为 json
def save_json(data, filename):
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {path}")


save_json(train, "train.json")
save_json(val, "val.json")
save_json(test, "test.json")

print("\n预处理完成！")
print("现在可以用 train.json / val.json / test.json 进行后续训练了。")