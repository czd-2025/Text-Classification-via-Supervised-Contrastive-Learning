import matplotlib.pyplot as plt
import numpy as np

# 数据
iters = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
         1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
train_loss = [2.8, 1.9, 1.4, 0.72, 1.3, 1.3, 0.97, 1.3, 0.76, 0.99, 0.97,
              0.71, 0.74, 0.95, 0.74, 0.66, 0.59, 0.5, 0.67, 0.43, 0.5]
train_acc = [0.00, 25.00, 56.25, 87.50, 43.75, 50.00, 68.75, 56.25, 75.00, 62.50, 68.75,
             68.75, 81.25, 68.75, 75.00, 68.75, 68.75, 81.25, 81.25, 81.25, 75.00]

# 创建图形和双Y轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制训练损失
color = 'tab:red'
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Train Loss', color=color, fontsize=12)
ax1.plot(iters, train_loss, color=color, linewidth=2, marker='o', markersize=6, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# 创建第二个Y轴
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Train Accuracy (%)', color=color, fontsize=12)
ax2.plot(iters, train_acc, color=color, linewidth=2, marker='s', markersize=6, label='Train Acc')
ax2.tick_params(axis='y', labelcolor=color)

# 设置标题
plt.title('Training Loss and Accuracy over Iterations', fontsize=14, fontweight='bold')

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

# 标注Epoch分界线
epoch_iters = [0, 800, 1600]
epoch_labels = ['Epoch 1', 'Epoch 2', 'Epoch 3']
for i, (iter_num, label) in enumerate(zip(epoch_iters[1:], epoch_labels[1:])):
    ax1.axvline(x=iter_num, color='gray', linestyle='--', alpha=0.5)
    ax1.text(iter_num + 50, ax1.get_ylim()[1] * 0.9, label, fontsize=10, color='gray')

plt.tight_layout()
plt.show()