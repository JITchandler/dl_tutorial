import torch

print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda:0')
    print(f"使用设备: {device}")
else:
    device = torch.device('cpu')
    print("使用CPU训练")