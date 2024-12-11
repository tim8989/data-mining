import torch
device = torch.device("mps" if torch.has_mps else "cpu")
print(device)
print(torch.backends.mps.is_available())  # 检查是否支持MPS
print(torch.__version__)