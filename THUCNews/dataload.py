import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('policy_data.csv')  # 假设数据保存在CSV文件中

# 假设文本列为 'text'，标签列为 'label'
texts = data['text'].values
labels = data['label'].values

# 划分数据集
train_size = 0.2
val_size = 0.1  # 最终验证集占总样本的10%
test_size = 0.1  # 最终测试集占总样本的10%

# 先进行训练集和临时集的划分
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=1 - train_size, random_state=42)

# 然后将临时集再次划分为验证集和测试集
val_share = val_size / (val_size + test_size)  # 计算验证集占临时集的比例
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_share, random_state=42)

print(f'Training samples: {len(X_train)}')
print(f'Validation samples: {len(X_val)}')
print(f'Testing samples: {len(X_test)}')

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 或者你使用的其他中文BERT模型

class BertPolicyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # 使用tokenizer进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'  # 返回PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    # 设置最大序列长度
    MAX_LENGTH = 128  # 根据你的数据调整

    # 创建数据集实例
    train_dataset = BertPolicyDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = BertPolicyDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = BertPolicyDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    # 创建数据加载器
    BATCH_SIZE = 32  # 根据你的硬件调整

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)