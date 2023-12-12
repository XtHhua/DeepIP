import os
import pickle

from typing import Dict
from torch import Tensor


class ModelIP:
    def __init__(self, model: str, dataset: str, method: str):
        self.model = model
        self.dataset = dataset
        self.method = method

    def save_ip(self, to_file: str, ip: Dict[str, Tensor]):
        if not os.path.exists(to_file):
            os.makedirs(os.path.dirname(to_file), exist_ok=True)
        #
        ip.update({"source_model": self.model})
        ip.update({"dataset": self.dataset})
        ip.update({"method": self.method})

        with open(to_file, mode="wb") as file:
            pickle.dump(obj=ip, file=file)
            print(f"save to {to_file} successfully!")

    def load_ip(self, from_file: str, ip: Dict[str, Tensor]):
        if not os.path.exists(from_file):
            raise FileNotFoundError(from_file)
        with open(from_file, "rb") as file:
            ip = pickle.load(file=file)
        return ip
