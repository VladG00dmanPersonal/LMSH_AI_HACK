from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path

class VSOSet(Dataset):
    def __init__(self, directory):
        directory = Path(directory)
        self.data = []
        for i in directory.iterdir():
            df = pd.read_csv(i)
            cols = []
            for i in 'ABCDEFGH':
                if i in df.columns:
                    cols.append(i)
                else:
                    for j in range(8):
                        if (i + str(j)) in df.columns:
                            cols.append(i + str(j))
            if len(cols) == 0:
                cols.append('total')
            for col in cols:
                for i in df.index:
                    for j in df.index:
                        if j < i:
                            self.data.append((df['name'][i], f['name'][j], df[col][i] > df[col][j]))
                        else: break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]