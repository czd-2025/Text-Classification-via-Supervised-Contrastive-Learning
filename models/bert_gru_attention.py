# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained import BertModel, BertTokenizer # <-- 【删除】旧的导入
from transformers import BertModel, BertTokenizer  # <-- 【新增】使用现代的 transformers 库


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        # 【说明】我将模型名称改回了原始名称，您可以在运行时看到SCL版本的模型名称是在Config中定义的
        self.model_name = 'bert-gru-attention-SCL'  # 在这里定义SCL版本，更清晰
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
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path, local_files_only=True)
        self.hidden_size = 768
        self.dropout = 0.6
        self.rnn_hidden = 768
        self.num_layers = 2

        ### === 新增：监督对比学习(SCL)相关超参数 === ###
        self.scl_temperature = 0.07
        self.lambda_scl = 0.1
        ### ======================================== ###


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, local_files_only=True)
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

        # 【说明】transformers库的BertModel默认不返回output_all_encoded_layers，所以可以简化调用
        # Bert返回的是一个元组，第一个元素是last_hidden_state
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