import torch
from torch.types import Device


def get_device() -> Device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'\"{device.type}\" is your training device.')
    return device
