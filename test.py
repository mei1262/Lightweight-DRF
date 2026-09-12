from collections import Counter

path = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single\labelResultAll.txt"
# 若第一行是表头 text,image 会自动跳过

name2id = {"negative": 0, "neutral": 1, "positive": 2}
id2name = {0: "negative", 1: "neutral", 2: "positive"}

N = 0
X = 0
# 清洗前：分别统计 text / image 标注
before_text = Counter()
before_image = Counter()
# 一致样本（将被保留）的标签
after_agree = Counter()

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 跳过表头
        if line.lower().startswith("text") or line.lower().startswith("id"):
            continue

        # 兼容: "1   neutral,positive" 或 "neutral,positive"
        if "," not in line:
            continue
        # 去掉行首编号
        if "\t" in line:
            body = line.split("\t")[-1].strip()
        else:
            parts = line.split(None, 1)
            body = parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else line

        text_lab, image_lab = [x.strip().lower() for x in body.split(",")]
        if text_lab not in name2id or image_lab not in name2id:
            continue

        N += 1
        before_text[text_lab] += 1
        before_image[image_lab] += 1

        if text_lab != image_lab:
            X += 1
        else:
            after_agree[text_lab] += 1

p = 100.0 * X / N if N else 0
print("N (清洗前) =", N)
print("X (冲突)   =", X)
print("p%         =", round(p, 2))
print("应保留     =", N - X)
print("before text:", dict(before_text))
print("before image:", dict(before_image))
print("agree kept:", dict(after_agree))
print("json 清洗后 = 4511；若 N-X 与 4511 接近则规则一致")