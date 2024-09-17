from dataclasses import dataclass
from typing import Tuple
import json


@dataclass
class TextConfig:
    def to_json(self, path: str):
        with open(path, "w") as fp:
            json.dump(self.__dict__, fp, indent=2)

    @classmethod
    def from_json(self, path: str):
        with open(path, "r") as fp:
            json_obj = json.load(fp)

        return TextConfig(**json_obj)

    # General
    project_name: str = "project_text"
    random_state: int = 42
    device: str = "cuda"
    seed: int = 42

    # Model
    num_classes: int = 4
    model_name: str = "roberta-base"
    pretrained: bool = True

    # Datase
    # # If fold = -1 no evaluation will be made and all the train dataset will be considered
    fold: int = 0
    csv_train_file: str = "dataset/train.csv"
    csv_test_file: str = "dataset/test.csv"
    csv_split_file: str = "dataset/fold.csv"
    batched: bool = True

    # Optimizer
    num_epochs: int = 3
    batch_size: int = 16
    test_batch_size: int = 16
    num_workers: int = 0

    lr: float = 2e-5
    weight_decay: float = 1e-2
