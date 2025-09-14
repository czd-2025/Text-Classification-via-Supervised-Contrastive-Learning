# Text-Classification-via-Supervised-Contrastive-Learning
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

中文文本分类，Bert及其变体，ERNIE，基于pytorch，开箱即用。

在上一个项目代码中加入监督对比学习模块，具体修改在train_eval.py中

## 介绍
模型介绍、数据流动过程：论文隐私

机器：4060 ， 训练时间：30分钟。  

## 监督对比学习损失函数

本节定义了一个函数 `scl_loss_fn`，用于计算监督对比学习 (Supervised Contrastive Learning, SCL) 损失。 此损失旨在拉近具有相同标签的样本的表示，同时推开具有不同标签的样本的表示。 它的目的是*在知识蒸馏损失的基础上*使用，允许模型既可以从教师模型的预测中学习，也可以从标记数据的底层结构中学习。

### 功能

`scl_loss_fn` 函数接收以下输入：

*   `features`: 一个 `[batch_size, feature_dim]` 的张量，包含模型输出的归一化特征向量（即表示）。 这些特征理想情况下应该是模型架构中某些转换的结果。
*   `labels`: 一个 `[batch_size]` 的张量，包含批次中每个样本的真实标签。
*   `temperature`: 一个浮点数，表示温度超参数。 此参数控制对比分布的锐度。 较高的温度会导致较柔和的概率。

该函数返回一个标量张量，表示计算出的 SCL 损失。

### 实现细节

该函数执行以下步骤：

1.  **标签掩码创建：** 创建一个形状为 `[batch_size, batch_size]` 的布尔掩码，其中如果 `labels[i]` 等于 `labels[j]`，则 `mask[i, j]` 为 `True`。 此掩码标识具有相同标签的样本对。

2.  **点积相似度计算：** 计算所有特征向量对之间的点积，并除以温度参数。 这会产生一个形状为 `[batch_size, batch_size]` 的相似度矩阵 `anchor_dot_contrast`，其中 `sim[i, j]` 表示 `features[i]` 和 `features[j]` 之间的相似度。

3.  **数值稳定性：** 从相似度矩阵中每行的最大值中减去该行的值，以提高指数运算期间的数值稳定性。

4.  **排除自我比较：** 创建一个掩码以排除每个样本与其自身进行比较（即，相似度矩阵的对角线元素）。 这是通过将 `[batch_size, batch_size]` 矩阵的对角线元素设置为 0 来实现的。然后将此掩码与标签掩码结合使用，以仅保留正样本对（不包括自我比较）。

5.  **对数概率计算：** 使用以下公式计算每个正样本对的对数概率：

    $log\_prob_{i,j} = sim_{i,j} - log(\sum_{k \neq i} exp(sim_{i,k}))$

    这涉及对 logits 进行指数运算、应用组合掩码和归一化。

6.  **SCL 损失计算：** SCL 损失计算为正样本对的对数概率的负平均值：

    $loss = - mean( \frac{\sum_{j} mask_{i,j} * log\_prob_{i,j}}{\sum_{j} mask_{i,j}} )$

    在样本在批次中没有正样本对的情况下，将一个小的常数 (`1e-9`) 添加到分母，以防止除以零。

### 与知识蒸馏集成

此 SCL 损失旨在与知识蒸馏损失结合使用。 那么，总体训练目标将是两种损失的加权组合：

$Loss_{total} = \alpha * Loss_{KD} + (1 - \alpha) * Loss_{SCL}$

其中：

*   $Loss_{KD}$ 是知识蒸馏损失。
*   $Loss_{SCL}$ 是由 `scl_loss_fn` 计算的监督对比学习损失。
*   $\alpha$ 是一个超参数，控制两种损失的相对重要性。 接近 1 的 $\alpha$ 值将优先考虑知识蒸馏，而接近 0 的值将优先考虑监督对比学习。

通过将知识蒸馏与监督对比学习相结合，学生模型可以从教师模型提供的软目标以及标记数据的底层结构中学习，从而可能提高性能和泛化能力。

预训练模型下载地址：  
bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz  
              词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  
来自[这里](https://github.com/huggingface/pytorch-transformers)   
备用：模型的网盘地址：https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw

ERNIE_Chinese: http://image.nghuyong.top/ERNIE.zip  
来自[这里](https://github.com/nghuyong/ERNIE-Pytorch)  
备用：网盘地址：https://pan.baidu.com/s/1lEPdDN1-YQJmKEd_g9rLgw  

解压后，按照上面说的放在对应目录下，文件名称确认无误即可。  

## 使用说明
下载好预训练模型就可以跑了。
```
# 训练并测试：
# bert
python run.py --model bert

# bert + 其它
python run.py --model bert_CNN

# ERNIE
python run.py --model ERNIE
```

### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  

## 未完待续
 - 封装预测功能


## 对应论文
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
[2] ERNIE: Enhanced Representation through Knowledge Integration  
