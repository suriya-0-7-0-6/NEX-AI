import torch
import os

def print_cuda_info():
    # Check if CUDA is available
    print("CUDA Available: ", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        # GPU details
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)} MB")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2)} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / (1024 ** 2)} MB")
        print(f"Current Device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available on this system.")

def print_pytorch_info():
    # PyTorch details
    print("\nPyTorch Version: ", torch.__version__)
    print("Is PyTorch built with CUDA: ", torch.cuda.is_available())
    print("CUDA Version used by PyTorch: ", torch.version.cuda)
    print("Device count: ", torch.cuda.device_count())

def print_system_info():
    # System info using nvidia-smi
    print("\nSystem GPU Info (using nvidia-smi):")
    os.system("nvidia-smi")

if __name__ == "__main__":
    print_cuda_info()
    print_pytorch_info()
    print_system_info()
