import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from dataset import MVSADataset
from models.baseline import BaselineModel
from models.lad_drf import LADDRF
from models.lightweight_drf_ablation import LightweightDRFAblation

data_root = r"C:\Users\梅煜寒\Desktop\Lightweight-DRF\data\MVSA_Single"
image_dir = os.path.join(data_root, "data")
test_json = os.path.join(data_root, "test.json")

CKPT_BASELINE = "best_baseline.pth"
CKPT_NO_RECOVERY = "best_ablation_no_recovery.pth"
CKPT_LAD = "best_lad_drf_c.pth"

BATCH_SIZE = 16
WARMUP_BATCHES = 8
MEASURE_BATCHES = 50


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_if_exists(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"  loaded: {path}")
    else:
        print(f"  WARNING: 未找到 {path}，使用随机初始化结构计参数/速度")
    return model


@torch.no_grad()
def measure_inference_time(model, loader, device, kind):
    model.eval()
    # warmup
    for i, batch in enumerate(loader):
        if i >= WARMUP_BATCHES:
            break
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if kind == "baseline":
            _ = model(images, input_ids, attention_mask)
        elif kind == "no_recovery":
            _ = model(images, input_ids, attention_mask, update_queue=False)
        else:
            _ = model(
                images, input_ids, attention_mask,
                update_queue=False, force_image_path=None,
            )

    times = []
    for i, batch in enumerate(loader):
        if i >= MEASURE_BATCHES:
            break
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if kind == "baseline":
            _ = model(images, input_ids, attention_mask)
        elif kind == "no_recovery":
            _ = model(images, input_ids, attention_mask, update_queue=False)
        else:
            _ = model(
                images, input_ids, attention_mask,
                update_queue=False, force_image_path=None,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_batch_ms = (sum(times) / len(times)) * 1000
    avg_sample_ms = avg_batch_ms / BATCH_SIZE
    return avg_batch_ms, avg_sample_ms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    loader = DataLoader(
        MVSADataset(test_json, image_dir, tokenizer, is_train=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    results = []

    # Baseline
    print("Baseline")
    baseline = BaselineModel(num_classes=3).to(device)
    load_if_exists(baseline, CKPT_BASELINE, device)
    b_total, b_train = count_parameters(baseline)
    b_batch, b_sample = measure_inference_time(baseline, loader, device, "baseline")
    results.append(("Baseline", b_total, b_train, b_batch, b_sample))

    # no_recovery
    print("no_recovery")
    no_rec = LightweightDRFAblation(
        num_classes=3, use_quality_weight=True, use_recovery=False,
    ).to(device)
    load_if_exists(no_rec, CKPT_NO_RECOVERY, device)
    n_total, n_train = count_parameters(no_rec)
    n_batch, n_sample = measure_inference_time(no_rec, loader, device, "no_recovery")
    results.append(("no_recovery", n_total, n_train, n_batch, n_sample))

    # LAD-DRF
    print("LAD-DRF")
    lad = LADDRF(num_classes=3).to(device)
    load_if_exists(lad, CKPT_LAD, device)
    l_total, l_train = count_parameters(lad)
    l_batch, l_sample = measure_inference_time(lad, loader, device, "lad")
    results.append(("LAD-DRF", l_total, l_train, l_batch, l_sample))

    print("\n" + "=" * 64)
    print("参数量对比")
    print("=" * 64)
    for name, total, train, _, _ in results:
        print(f"{name:<12}: {total/1e6:.2f} M  (trainable {train/1e6:.2f} M)")
    print(f"{'原版DRF估算':<12}: ≈200 M  (ViT-B + BERT-base)")

    print("\n" + "=" * 64)
    print(f"推理速度 (batch_size={BATCH_SIZE}, warmup={WARMUP_BATCHES})")
    print("=" * 64)
    for name, _, _, batch_ms, sample_ms in results:
        print(f"{name:<12}: {batch_ms:.2f} ms/batch  |  {sample_ms:.2f} ms/sample")
    print("=" * 64)


if __name__ == "__main__":
    main()