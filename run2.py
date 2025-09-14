# coding: UTF-8
import time
import torch
import numpy as np
# === 【关键修正】我们只导入 train 函数，因为 init_network 不再需要 ===
from train_eval import train
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# 注意：您的命令行参数是 '--model 1'，所以这里我将 help 信息改得更通用一些
parser.add_argument('--model', type=str, required=True,
                    help='choose a model file name from the models folder, e.g., bert_gru_attention1')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 这部分代码完全正确，无需改动
    model_name = args.model
    # 当您输入 --model 1 时，这里会报错，因为没有 models.1.py 文件
    # 请确保您传入的是正确的文件名，例如我们一直在用的 'bert_gru_attention1'
    # 我假设您在运行时会使用正确的名称，例如：--model bert_gru_attention1
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config, shuffle=True)
    dev_iter = build_iterator(dev_data, config, shuffle=False)
    test_iter = build_iterator(test_data, config, shuffle=False)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # 这两行代码也完全正确，无需改动
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
