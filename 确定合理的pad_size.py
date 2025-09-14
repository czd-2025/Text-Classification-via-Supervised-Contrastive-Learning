import numpy as np
# 确保我们使用的是您项目里对应的库
from pytorch_pretrained_bert import BertTokenizer

# --- 配置区 ---
# 确保这个路径指向你项目中的训练文件
txt_file_path = './THUCNews/data/train.txt'

# 确保这个路径指向你下载的预训练模型文件夹
bert_path = './bert_pretrain'


# --- 结束配置 ---


def analyze_text_lengths(file_path, tokenizer):
    """读取txt文件，分析每行文本的token长度"""
    token_lengths = []
    print(f"正在打开文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='UTF-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t', 1)
                if len(parts) >= 1:
                    text = parts[0]

                    # --- 这里是关键的修改 ---
                    # 1. 使用正确的 .tokenize() 方法
                    tokens = tokenizer.tokenize(text)

                    # 2. 计算长度时要加上2
                    # 因为BERT模型在处理时，会在句子开头加上[CLS]，结尾加上[SEP]两个特殊token
                    # 所以实际输入模型的长度是 token数量 + 2
                    length = len(tokens) + 2
                    token_lengths.append(length)

    except FileNotFoundError:
        print(f"错误：找不到文件！请检查路径是否正确: {file_path}")
        return None
    except Exception as e:
        print(f"读取或处理文件时发生错误: {e}")
        return None

    return np.array(token_lengths)


if __name__ == '__main__':
    print("正在加载BERT分词器...")
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)

    lengths_np = analyze_text_lengths(txt_file_path, bert_tokenizer)

    if lengths_np is not None and len(lengths_np) > 0:
        print("\n--- 文本Token长度分析报告 (已修正) ---")
        print(f"已分析 {len(lengths_np)} 条文本。")
        print(f"最短文本Token长度: {np.min(lengths_np)}")
        print(f"最长文本Token长度: {np.max(lengths_np)}")
        print(f"平均文本Token长度: {np.mean(lengths_np):.2f}")
        print(f"文本Token长度中位数: {np.median(lengths_np):.0f}")
        print("-" * 30)
        # 计算不同百分位数的文本长度
        print(f"80%的文本长度小于: {np.percentile(lengths_np, 80):.0f}")
        print(f"90%的文本长度小于: {np.percentile(lengths_np, 90):.0f}")
        print(f"95%的文本长度小于: {np.percentile(lengths_np, 95):.0f}")
        print(f"98%的文本长度小于: {np.percentile(lengths_np, 98):.0f}")
        print(f"最长的文本长度(100%): {np.percentile(lengths_np, 100):.0f}")
        print("-" * 30)
        print("\n【决策建议】")
        print("1. 观察'95%的文本长度小于'这一行的数值。这是一个非常好的 `pad_size` 起始点。")
        print("2. 选择一个比这个数值稍大一点的、比较“整”的数字（最好是2的幂，如64, 128, 256）。")
        print("3. 例如：如果95%的文本长度小于110，那么选择 `pad_size = 128` 是一个非常理想的选择。")