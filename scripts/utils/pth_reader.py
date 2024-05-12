import torch
import sys

file_path = sys.argv[1]
data = torch.load(file_path)

print("Content:")
for key, value in data.items():
    print(f"{key}: {value}")