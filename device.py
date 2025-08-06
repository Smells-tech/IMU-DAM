import torch

# ------------------------------------------------------------------------------------------------------
#   Print CUDA status
# ------------------------------------------------------------------------------------------------------

cuda_enabled = torch.cuda.is_available()
if cuda_enabled:
    print(f"\nCUDA is supported by this system? {cuda_enabled}")
    print(f"CUDA version: {torch.version.cuda}")
    # Storing ID of current CUDA device
    device = 'cuda'
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
    # torch.backends.cuda.matmul.allow_tf32 = True
    print(f"allow tf32: {torch.backends.cuda.matmul.allow_tf32}\n")
else:
    device = 'cpu'
