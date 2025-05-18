import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. Possible reasons:")
    print("1. NVIDIA GPU drivers are not installed or outdated")
    print("2. PyTorch was installed without CUDA support")
    print("3. Your GPU is not CUDA-compatible")