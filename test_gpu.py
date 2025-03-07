import torch
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"當前使用的 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA 不可用")
