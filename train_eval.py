# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
# === 【关键修改】导入您指定的 BertAdam 优化器 ===
from pytorch_pretrained_bert.optimization import BertAdam


# 定义SCL Loss的辅助函数
def scl_loss_fn(features, labels, temperature):
    """
    计算Supervised Contrastive Loss
    输入:
        features: [batch_size, feature_dim] - 归一化的特征向量 (来自模型输出的 representation)
        labels: [batch_size] - 真实的标签
        temperature: float - 温度超参数
    输出:
        loss: 一个标量张量
    """
    device = features.device
    batch_size = features.shape[0]

    # 1. 创建标签掩码 (mask)
    # labels.unsqueeze(1) == labels.unsqueeze(0) 会创建一个 [batch_size, batch_size] 的布尔矩阵
    # 其中 mask[i, j] 在 labels[i] == labels[j] 时为 True
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # 2. 计算点积相似度
    # anchor_dot_contrast 的形状为 [batch_size, batch_size]
    # sim[i, j] = features[i] · features[j]
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T),
        temperature)

    # 3. 为了数值稳定，减去每行的最大值
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # 4. 创建一个对角线为0的掩码，以排除自己和自己的比较 (i.e., logit_ii)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    # 将原始的标签掩码和对角线掩码结合，只保留正样本对(不包括自己)
    mask = mask * logits_mask

    # 5. 计算 log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # 6. 计算SCL Loss
    # mask.sum(1) 是每个样本的正样本对数量
    # 我们只在有正样本对的情况下计算损失 (避免除以0)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

    # 取所有样本损失的平均值
    loss = - mean_log_prob_pos.mean()

    return loss


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()

    # === 【关键修改】完全按照您的知识蒸馏代码，配置BertAdam优化器 ===
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    # ===============================================================

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            # 模型的forward现在返回两个值
            logits, representation = model(trains)

            # --- 损失计算 ---
            # 1. 标准的分类损失 (Cross Entropy)
            loss_ce = F.cross_entropy(logits, labels)

            # 2. 监督对比学习损失 (SCL)
            loss_scl = scl_loss_fn(representation, labels, config.scl_temperature)

            # 3. 组合总损失
            # 使用lambda_scl作为权重来平衡两种损失
            loss = loss_ce + config.lambda_scl * loss_scl

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(logits.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                # 更新打印信息，方便调试
                msg = 'Iter: {0:>6},  Total Loss: {1:>5.2f},  CE Loss: {2:>5.2f},  SCL Loss: {3:>5.2f},  Train Acc: {4:>6.2%},  Val Loss: {5:>5.2f},  Val Acc: {6:>6.2%},  Time: {7} {8}'
                print(
                    msg.format(total_batch, loss.item(), loss_ce.item(), loss_scl.item(), train_acc, dev_loss, dev_acc,
                               time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2f},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            logits, _ = model(texts)  # 评估时，我们只需要logits
            loss = F.cross_entropy(logits, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(logits.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)