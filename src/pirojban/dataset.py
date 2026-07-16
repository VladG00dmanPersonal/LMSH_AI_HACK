from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path


class VSOSet(Dataset):
    def __init__(self, directory):
        directory = Path(directory)
        self.data = []
        d1 = pd.read_csv(directory / "belchonok.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

