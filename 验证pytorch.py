import torch

# 检查 CUDA 是否可用，这是前提
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查 PyTorch 是否启用了 cuDNN
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

# 打印 PyTorch 当前正在使用的 cuDNN 版本号
# 这将直接反映你替换进去的那个版本！
if torch.cuda.is_available():
    print(f"cuDNN version: {torch.backends.cudnn.version()}")