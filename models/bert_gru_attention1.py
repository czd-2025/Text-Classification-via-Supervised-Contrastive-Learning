# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
# ===【新增导入】我们需要BertConfig来手动加载配置 ===
from transformers import BertModel, BertTokenizer, BertConfig


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert-gru-attention-SCL'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 2000
        self.num_classes = len(self.class_list)
        self.num_epochs = 8
        self.batch_size = 16
        self.pad_size = 64
        self.learning_rate = 1e-5
        self.bert_path = './bert_pretrain'

        # 这一行保持原样，因为它没有引发问题
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.hidden_size = 768
        self.dropout = 0.6
        self.rnn_hidden = 768
        self.num_layers = 2

        self.scl_temperature = 0.07
        self.lambda_scl = 0.1


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # === 【最终决战：手动分步加载，彻底杜绝尺寸错误】 ===

        # 第1步：只加载config.json文件，创建一个配置对象
        print("====> Step 1: Explicitly loading configuration from local config.json...")
        bert_config = BertConfig.from_pretrained(config.bert_path)

        # 第2步：打印出我们加载到的词汇表大小，进行最终验证
        print(f"====> Step 2: Loaded config. Vocab size is: {bert_config.vocab_size}")
        if bert_config.vocab_size != 21128:
            print("!!!! CRITICAL ERROR: Loaded config vocab size is NOT 21128 !!!!")

        # 第3步：使用这个我们100%确认的配置对象，来加载模型权重
        # 这个调用会先用 bert_config 创建正确尺寸的空模型，再填入权重
        print("====> Step 3: Creating model from this explicit config and loading weights...")
        self.bert = BertModel.from_pretrained(config.bert_path, config=bert_config)
        print("====> Step 4: Model loading process finished.")

        # =======================================================

        for param in self.bert.parameters():
            param.requires_grad = True

        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.rnn_hidden,
            num_layers=config.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        self.attention_W = nn.Linear(config.rnn_hidden * 2, config.rnn_hidden * 2)
        self.attention_v = nn.Parameter(torch.FloatTensor(config.rnn_hidden * 2, 1))
        nn.init.xavier_uniform_(self.attention_v)

        self.dropout_layer = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]

        outputs = self.bert(context, attention_mask=mask)
        encoder_out = outputs.last_hidden_state

        gru_out, _ = self.gru(encoder_out)

        u = torch.tanh(self.attention_W(gru_out))
        scores = u.matmul(self.attention_v).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=1)

        representation = torch.sum(attention_weights.unsqueeze(-1) * gru_out, dim=1)

        out = self.dropout_layer(representation)
        logits = self.fc(out)

        normalized_representation = F.normalize(representation, p=2, dim=1)

        return logits, normalized_representation