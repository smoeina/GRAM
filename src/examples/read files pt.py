import torch
from torch_geometric.data import Data

# ---------- SETTINGS ----------
file_path = r"C:\Users\USER\PycharmProjects\GARM - Output Classification\data\processed\pre_transform.pt"
# ------------------------------

# Allowlist torch_geometric Data type so torch.load can read it
torch.serialization.add_safe_globals([Data])

# Load file (safe mode: weights_only=True by default)
# Set weights_only=False if you need to load arbitrary objects and TRUST the file
data = torch.load(file_path, map_location="cpu", weights_only=False)

def print_structure(obj, indent=0, max_tensor_elements=10):
    """Recursively prints the structure of an object."""
    prefix = " " * indent

    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k}: {type(v)}")
            print_structure(v, indent + 2, max_tensor_elements)

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            print(f"{prefix}[{i}]: {type(v)}")
            print_structure(v, indent + 2, max_tensor_elements)

    elif torch.is_tensor(obj):
        # Print shape and optionally a small preview
        print(f"{prefix}Tensor shape: {tuple(obj.shape)}")
        numel = obj.numel()
        if numel <= max_tensor_elements:
            print(f"{prefix}{obj}")
        else:
            print(f"{prefix}Preview: {obj.flatten()[:max_tensor_elements]} ...")

    else:
        print(f"{prefix}{repr(obj)}")

# Show type and full nested structure
print("Top-level type:", type(data))
print("\nStructure of file content:\n")
print_structure(data)
