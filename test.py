import torch

# Check if PyTorch is installed correctly
print("PyTorch version:", torch.__version__)

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Check the number of available GPUs
print("Number of GPUs:", torch.cuda.device_count())

# Check the current CUDA device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Current CUDA device:", torch.cuda.get_device_name(device))
else:
    print("No CUDA devices found.")

# Create a tensor and move it to the GPU (if available)
tensor = torch.randn(1024, 1024)
if torch.cuda.is_available():
    tensor = tensor.to(device)

print("Tensor created successfully on", tensor.device)

