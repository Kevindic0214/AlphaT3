"""Script with helper functions."""
import warnings
import functools
import pathlib
import random

import numpy
import torch


def set_random_seed(seed: int = 0) -> None:
    """Sets random seed."""
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多個 GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model: torch.nn.Module, model_name: str, args) -> None:
    """Saves model checkpoint.

    Uses torch.save() to save PyTorch models.

    Args:
        model: PyTorch model.
        model_name: Name of policy model.
        args: Parsed arguments.
    """
    checkpoint_name = f"{f'{model_name}_{args.algorithm}' if model_name else 'model'}"
    checkpoint_path = "weights"
    model_path = pathlib.Path(checkpoint_path) / f"{checkpoint_name}.pth"

    # 確保目錄存在
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    torch.save(obj=model.state_dict(), f=model_path)
    print(f"\nModel '{checkpoint_name}' saved to {model_path}\n")


def load_checkpoint(model: torch.nn.Module, args) -> None:
    """Loads model from checkpoint.

    Args:
        model: PyTorch model.
        model_name: Name of policy model.
        args: Parsed arguments.
    """
    checkpoint_name = f"{f'{args.model_name}_{args.algorithm}' if args.model_name else 'model'}"
    checkpoint_path = "weights"
    model_path = pathlib.Path(checkpoint_path) / f"{checkpoint_name}.pth"

    if model_path.is_file():
        # 根據當前設備加載模型
        if torch.cuda.is_available():
            state_dict = torch.load(f=model_path, map_location=torch.device('cuda'))
        else:
            state_dict = torch.load(f=model_path, map_location=torch.device('cpu'))
            
        model.load_state_dict(state_dict=state_dict)
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"\nModel '{checkpoint_name}' loaded on {device_name}.\n")
    else:
        warnings.warn(f"\nModel checkpoint '{checkpoint_name}' not found. " "Continuing with random weights.\n")


def print_args(args) -> None:
    """Prints parsed arguments to console.

    Args:
        args: Parsed arguments.
    """
    print("Configuration:\n")
    representation = "{k:.<32}{v}"
    for key, value in vars(args).items():
        print(representation.format(k=key, v=value))
    print()
    
    # 顯示 CUDA 資訊
    if torch.cuda.is_available():
        print(f"CUDA 可用: 使用 {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"當前設備: GPU")
    else:
        print("CUDA 不可用: 使用 CPU")
    print()
