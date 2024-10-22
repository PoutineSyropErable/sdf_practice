import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU Device Count:", torch.cuda.device_count())
print("Current GPU Device:", torch.cuda.current_device())
print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))


print("ROCm available:", torch.backends.mps.is_available())  # Check for ROCm support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example to test tensor movement to the GPU
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print("Tensor on device:", x.device)
