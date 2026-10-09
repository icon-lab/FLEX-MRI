"""
Device and checkpoint helpers.
"""

import torch as th


def dev():
    """
    Get the device to train on.
    """
    if th.cuda.is_available():
        return th.device("cuda:0")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch checkpoint (a state dict) from ``path``.
    """
    return th.load(path, **kwargs)
