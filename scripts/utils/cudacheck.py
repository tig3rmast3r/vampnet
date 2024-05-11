import torch

cuda_available = torch.cuda.is_available()

print("CUDA available:", cuda_available)

if cuda_available:
    num_gpus = torch.cuda.device_count()
    print("GPUs:", num_gpus)
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not Available")
