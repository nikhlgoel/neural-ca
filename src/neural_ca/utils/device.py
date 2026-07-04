"""Device resolution honouring the config's ``device`` field."""

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Map ``'auto' | 'cpu' | 'cuda'`` to a concrete ``torch.device``."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
