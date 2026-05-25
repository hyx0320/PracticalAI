import torch
print("PyTorch版本:", torch.__version__)  # 看输出是否包含 '+cu130'，如果不包含，说明装了CPU版本。
print("CUDA可用:", torch.cuda.is_available())
