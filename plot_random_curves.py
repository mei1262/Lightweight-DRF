import os
import matplotlib.pyplot as plt
import numpy as np

# ========== 表 4-3 数据（Acc mean）==========
ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

# 类型 D（丢弃）——主图常用，曲线最清晰
baseline_d = [0.7087, 0.7006, 0.6792, 0.6593, 0.6431]
no_recovery_d = [0.7139, 0.7035, 0.6822, 0.6674, 0.6549]
lad_d = [0.7257, 0.7146, 0.6962, 0.6807, 0.6645]

# 可选：C+D
baseline_cd = [0.7094, 0.6991, 0.6807, 0.6630, 0.6504]
no_recovery_cd = [0.7161, 0.7065, 0.6895, 0.6785, 0.6681]
lad_cd = [0.7286, 0.7176, 0.7013, 0.6888, 0.6733]

out_dir = "results/figures"
os.makedirs(out_dir, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# ----- 图：D 设定 -----
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(ratios, baseline_d, "o--", label="Baseline", linewidth=1.8, markersize=6)
ax.plot(ratios, no_recovery_d, "s--", label="no_recovery", linewidth=1.8, markersize=6)
ax.plot(ratios, lad_d, "D-", label="LAD-DRF", linewidth=2.0, markersize=6)

ax.set_xlabel("Disruption ratio")
ax.set_ylabel("Accuracy")
ax.set_title("Random disruption (Type D)")
ax.set_xticks(ratios)
ax.set_ylim(0.62, 0.75)
ax.grid(True, linestyle=":", alpha=0.7)
ax.legend(frameon=True)
fig.tight_layout()
path_d = os.path.join(out_dir, "fig4_1_random_D.png")
fig.savefig(path_d, bbox_inches="tight")
fig.savefig(path_d.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved: {path_d}")
plt.close()

# ----- 可选：C+D -----
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(ratios, baseline_cd, "o--", label="Baseline", linewidth=1.8, markersize=6)
ax.plot(ratios, no_recovery_cd, "s--", label="no_recovery", linewidth=1.8, markersize=6)
ax.plot(ratios, lad_cd, "D-", label="LAD-DRF", linewidth=2.0, markersize=6)
ax.set_xlabel("Disruption ratio")
ax.set_ylabel("Accuracy")
ax.set_title("Random disruption (Type C+D)")
ax.set_xticks(ratios)
ax.set_ylim(0.62, 0.75)
ax.grid(True, linestyle=":", alpha=0.7)
ax.legend(frameon=True)
fig.tight_layout()
path_cd = os.path.join(out_dir, "fig4_1_random_CD.png")
fig.savefig(path_cd, bbox_inches="tight")
fig.savefig(path_cd.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved: {path_cd}")
plt.close()

print("Done. 论文主图建议用 Type D 的 PNG/PDF。")