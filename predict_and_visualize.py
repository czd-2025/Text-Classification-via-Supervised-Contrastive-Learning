# predict_and_visualize.py

import torch
from models.bert_gru_attention import Model, Config  # 确保从正确的路径导入你的模型
from visualize_util import show_attention_heatmap  # 导入我们刚创建的函数


def predict_single_sentence(model, config, text):
    """
    对单个句子进行预测并可视化其注意力权重。
    """
    # 将模型设置为评估模式
    model.eval()

    # 1. 文本预处理
    tokens = config.tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    # 截断或填充到pad_size
    if len(tokens) < config.pad_size:
        seq_len = len(tokens)
        mask = [1] * seq_len + [0] * (config.pad_size - seq_len)
        token_ids = config.tokenizer.convert_tokens_to_ids(tokens) + [0] * (config.pad_size - seq_len)
    else:
        seq_len = config.pad_size
        mask = [1] * config.pad_size
        tokens = tokens[:config.pad_size]
        token_ids = config.tokenizer.convert_tokens_to_ids(tokens)

    # 转换为Tensor
    token_ids_tensor = torch.LongTensor([token_ids]).to(config.device)
    mask_tensor = torch.LongTensor([mask]).to(config.device)
    # 模型输入是元组 (ids, seq_len, mask)
    # 注意：这里的seq_len参数在你的模型中没有被使用，但为了保持项目统一性，我们传入
    inputs = (token_ids_tensor, None, mask_tensor)

    # 2. 模型预测，并请求返回Attention
    with torch.no_grad():
        # 调用forward时，设置 return_attention=True
        outputs, attention_weights = model(inputs, return_attention=True)

    # 3. 解析预测结果
    pred_class_idx = torch.argmax(outputs).item()
    pred_class = config.class_list[pred_class_idx]

    print("=" * 30)
    print(f"输入句子: {text}")
    print(f"预测类别: {pred_class}")
    print("=" * 30)

    # 4. 可视化Attention
    # attention_weights 的 shape 是 [1, pad_size]
    # 我们只可视化有效长度部分的权重
    valid_weights = attention_weights[0, :seq_len]

    # 为了让可视化更清晰，通常不显示 [CLS] 和 [SEP]
    tokens_to_show = tokens[1:-1]
    weights_to_show = valid_weights[1:-1]

    show_attention_heatmap(tokens_to_show, weights_to_show, title="句子注意力权重分布")


if __name__ == '__main__':
    # --- 配置 ---
    dataset = 'THUCNews'  # 你的数据集名称
    config = Config(dataset)
    model = Model(config).to(config.device)

    # --- 加载已训练好的模型 ---
    # 确保你的模型已经训练并保存在了正确的路径
    try:
        model.load_state_dict(torch.load(config.save_path, map_location=config.device))
        print("模型加载成功！")
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {config.save_path}")
        print("请先运行 train_eval.py 训练模型。")
        exit()

    # --- 输入你想测试的句子 ---
    sentence1 = "对首次认定的自治区级大数据骨干企业、示范企业、优秀应用解决方案企业，分别给予50万元、30万元、10万元的一次性奖励"
    sentence2 = "新款iPhone发布，苹果公司股价大幅上涨。"

    predict_single_sentence(model, config, sentence1)
    predict_single_sentence(model, config, sentence2)