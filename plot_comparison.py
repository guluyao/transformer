import matplotlib.pyplot as plt
import pandas as pd
import os

# 实验CSV绝对路径
experiments = {
    "D:\\pycharm\\transformer-midterm\\results\\baseline\\training_results.csv":
        ("#1f77b4", "基线模型（全组件）", "-"),  # 蓝色，实线

    "D:\\pycharm\\transformer-midterm\\results\\ablate_pe\\training_results.csv":
        ("#ff0000", "消融位置编码", "--"),  # 红色，虚线

    "D:\\pycharm\\transformer-midterm\\results\\ablate_multihead\\training_results.csv":
        ("#ff8800", "消融多头注意力（单头）", ":"),  # 橙色，点线

    "D:\\pycharm\\transformer-midterm\\results\\ablate_ffn\\training_results.csv":
        ("#33b5e5", "消融FFN", "-."),  # 青色，点划线
}

# 绘图样式配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.labelsize'] = 11

# 读取并整理所有数据
all_data = []
train_loss_ranges = {}
val_loss_ranges = {}

for csv_path, (color, label, linestyle) in experiments.items():
    if not os.path.exists(csv_path):
        print(f"⚠️  警告：{csv_path} 不存在！")
        continue

    try:
        df = pd.read_csv(csv_path)
        epochs = df["epoch"].values
        train_loss = df["train_loss"].values
        val_loss = df["val_loss"].values
        all_data.append((epochs, train_loss, val_loss, color, label, linestyle))

        # 记录损失范围
        train_loss_ranges[label] = (train_loss.min(), train_loss.max())
        val_loss_valid = val_loss[~pd.isna(val_loss)]
        val_loss_ranges[label] = (val_loss_valid.min(), val_loss_valid.max()) if len(val_loss_valid) > 0 else (0, 0)

        print(
            f"✅ {label}：训练损失 {train_loss.min():.2f}~{train_loss.max():.2f}，验证损失 {val_loss_ranges[label][0]:.2f}~{val_loss_ranges[label][1]:.2f}")
    except Exception as e:
        print(f"❌ {label} 读取失败：{str(e)}")
        continue

# 计算y轴全局范围
train_min = min([v[0] for v in train_loss_ranges.values()])
train_max = max([v[1] for v in train_loss_ranges.values()])
val_min = min([v[0] for v in val_loss_ranges.values()])
val_max = max([v[1] for v in val_loss_ranges.values()])

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# 绘制所有曲线（按线条样式区分）
for data in all_data:
    epochs, train_loss, val_loss, color, label, linestyle = data
    # 训练损失曲线
    ax1.plot(epochs, train_loss, color=color, label=label, marker='o', markersize=4,
             linestyle=linestyle, alpha=0.8)
    # 验证损失曲线
    val_valid = ~pd.isna(val_loss)
    if val_valid.any():
        ax2.plot(epochs[val_valid], val_loss[val_valid], color=color, label=label, marker='s', markersize=4,
                 linestyle=linestyle, alpha=0.8)

# 设置y轴范围
ax1.set_ylim(bottom=train_min - 0.2, top=train_max + 0.2)
ax2.set_ylim(bottom=val_min - 0.2, top=val_max + 0.2)

# 美化子图
ax1.set_xlabel("训练轮次（Epoch）")
ax1.set_ylabel("训练损失（Cross-Entropy Loss）")
ax1.set_title("各实验训练损失对比", fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')

ax2.set_xlabel("训练轮次（Epoch）")
ax2.set_ylabel("验证损失（Cross-Entropy Loss）")
ax2.set_title("各实验验证损失对比", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# 保存图片
save_path = "消融实验对比图.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 对比图已保存至：{os.path.abspath(save_path)}")
plt.show()