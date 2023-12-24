import time
import os
import random

import torch
import numpy as np


def timer(func: callable) -> callable:
    """A timer decorator to time the execution of a function.

    Args:
        func (_type_): Functions that require timing.
    Returns:
        function: The decorated function.
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end - start):.2f} seconds to execute.")
        return result

    return wrapper


def seed_everything(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
