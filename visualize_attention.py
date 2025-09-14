# visualize_util.py

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def show_attention_heatmap(tokens, weights, title="Attention Weights Heatmap"):
    """
    绘制注意力权重的热力图。

    参数:
    - tokens (list of str): 经过分词后的token列表。
    - weights (torch.Tensor or numpy.array): 对应每个token的注意力权重，应为1D张量或数组。
    - title (str): 图表的标题。
    """
    # 确保权重数据在CPU上，并且是numpy array格式
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().detach().numpy()

    # 创建一个figure和axes对象，并设置大小
    # 图的宽度与token数量成正比，高度固定，以获得最佳视觉效果
    fig, ax = plt.subplots(figsize=(len(tokens) * 0.8, 2.5))

    # 使用seaborn绘制热力图
    # 注意：weights需要是2D的，所以我们用[]将其包裹起来
    sns.heatmap(
        [weights],
        xticklabels=tokens,  # X轴标签设置为我们的token
        yticklabels=False,  # Y轴不需要标签
        cmap="viridis",  # 选择一个好看的颜色主题，"Reds", "Blues" 也很常用
        annot=True,  # 在每个格子上显示权重数值
        fmt=".3f",  # 数值格式化为3位小数
        cbar=False,  # 可以隐藏或显示颜色条
        ax=ax
    )

    # 解决中文显示问题，你需要确保你的系统中有'SimHei'这个字体
    # 如果没有，可以换成 'Microsoft YaHei' (微软雅黑) 或其他你已安装的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    ax.set_title(title, fontsize=16, pad=20)  # 设置标题和边距
    plt.xticks(rotation=45)  # X轴的标签旋转45度，防止文字重叠
    plt.tight_layout()  # 自动调整布局
    plt.show()
