# coding: UTF-8
import torch
import torch.nn as nn
# 注意：虽然此文件没有直接使用 F 中的函数，但为保持和 bert_CNN.py 的风格一致性，我们保留此导入。
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class Config(object):
    """
    配置参数 (严格仿照 bert_CNN.py 的结构)
    """

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 99999                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 8                                             # epoch数
        self.batch_size = 8                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 3e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 128
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 128                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 64
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # model_config = BertConfig.from_pretrained(config.bert_path)
        #
        # # 使用预训练模型的配置加载 BERT 模型
        # self.bert = BertModel.from_pretrained(
        #     config.bert_path,
        #     config=model_config  # 显式指定配置，确保 vocab_size 正确
        # )
        # 设置 BERT 的所有参数都参与梯度计算和更新
        for param in self.bert.parameters():
            param.requires_grad = True
        # 定义 GRU 层
        self.gru = nn.GRU(
            input_size=config.hidden_size,  # 输入维度: 来自BERT的输出
            hidden_size=config.rnn_hidden,  # GRU隐藏层维度
            num_layers=config.num_layers,  # GRU层数
            bidirectional=True,  # 使用双向GRU
            batch_first=True,  # 输入数据的第一维是batch
            dropout=config.dropout if config.num_layers > 1 else 0  # 层间dropout
        )

        self.dropout_layer = nn.Dropout(config.dropout)

        # 定义最后的全连接分类层
        # 输入维度是 GRU 隐藏层维度的两倍，因为是双向的 (前向 + 后向)
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        # x 是一个元组，包含: (句子的id, 句子长度, attention_mask)
        context = x[0]  # 输入的句子 tensor, shape: [batch_size, pad_size]
        mask = x[2]  # 对padding部分进行mask, shape: [batch_size, pad_size]

        # 通过 BERT 模型得到编码输出
        # encoder_out shape: [batch_size, pad_size, hidden_size]
        encoder_out, _ = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)

        # 将 BERT 的输出直接送入 GRU
        # gru_out shape: [batch_size, pad_size, rnn_hidden * 2]
        # _ (final_hidden_state) shape: [num_layers * 2, batch_size, rnn_hidden]
        gru_out, _ = self.gru(encoder_out)

        # 为了分类，我们通常取序列最后的隐藏状态
        # gru_out[:, -1, :] 提取了每个样本序列中最后一个时间步的输出
        # shape: [batch_size, rnn_hidden * 2]
        last_hidden_state = gru_out[:, -1, :]

        # 应用 dropout 后送入全连接层
        out = self.dropout_layer(last_hidden_state)
        out = self.fc(out)
        return out