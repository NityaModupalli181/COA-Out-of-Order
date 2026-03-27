# import torch
# print(torch.cuda.is_available())          # Should be True
# print(torch.cuda.get_device_name(0))      # Your GPU name
# print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB") 
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (PyTorch):", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())