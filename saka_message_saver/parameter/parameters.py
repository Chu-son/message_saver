import dataclasses
import yaml

from typing import List


@dataclasses.dataclass
class Parameters:
    ROI: List[int] = None

    def load_from_yaml(self, filename: str):
        with open(filename, 'r') as f:
            params = yaml.safe_load(f)

        self.ROI = params['ROI']

    def save_to_yaml(self, filename: str):
        with open(filename, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f)
