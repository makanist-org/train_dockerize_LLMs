import torch
import platform

def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"System: {platform.system()} {platform.machine()}")
    
    # Check for CUDA (NVIDIA GPUs)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check for MPS (Apple Silicon)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS (Apple Metal) available: {mps_available}")
    
    # Determine device
    if cuda_available:
        device = "cuda"
    elif mps_available:
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    if not cuda_available and not mps_available:
        print("\nNo GPU acceleration available. Possible reasons:")
        if platform.system() == "Darwin" and "arm64" in platform.machine():
            print("- You're using Apple Silicon but MPS might not be enabled in PyTorch")
            print("- Try updating PyTorch: pip install --upgrade torch")
        elif platform.system() == "Darwin":
            print("- MacBooks with Intel processors use AMD GPUs, not NVIDIA")
            print("- CUDA is not supported on Mac")
        else:
            print("1. NVIDIA GPU drivers are not installed or outdated")
            print("2. PyTorch was installed without CUDA support")
            print("3. Your GPU is not CUDA-compatible")
    
    return device

if __name__ == "__main__":
    device = check_gpu()