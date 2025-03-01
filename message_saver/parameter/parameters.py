import os
import dataclasses
import yaml

from typing import List

import message_saver


@dataclasses.dataclass
class Parameters:
    ROI: List[int] = None
    base_directory: str = os.path.join(
        message_saver.PROJECT_ROOT_PATH, 'images')
    directory: str = base_directory
    sub_directory: str = "test"
    filename_base: str = ""

    loop_times: int = -1
    reverse: bool = False
    start_index: int = 0

    @classmethod
    def load_from_yaml(cls, filename: str):
        with open(filename, 'r') as f:
            params = yaml.safe_load(f)

        return cls(**params)

    def save_to_yaml(self, filename: str):
        with open(filename, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f)
