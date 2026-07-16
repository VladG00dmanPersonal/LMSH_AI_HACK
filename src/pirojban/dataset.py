import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VSOSet(Dataset):
    def __init__(self, data: pd.DataFrame, seed=42):
        super().__init__()
        self.data = data.copy()
        self.user_to_id = {
            user: user_id
            for user_id, user in enumerate(self.data.index)
        }
        self.id_to_user = {
            user_id: user
            for user, user_id in self.user_to_id.items()
        }
        self.task_to_id = {
            task: task_id
            for task_id, task in enumerate(self.data.columns)
        }
        self.id_to_task = {
            task_id: task
            for task, task_id in self.task_to_id.items()
        }
        rng = np.random.default_rng(seed)
        athlete_a_ids = []
        athlete_b_ids = []
        task_ids = []
        targets = []
        for task, task_id in self.task_to_id.items():
            scores = self.data[task].dropna()
            users = scores.index.to_numpy()
            values = scores.to_numpy(dtype=np.float32)
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    score_i = values[i]
                    score_j = values[j]
                    if score_i == score_j:
                        continue
                    user_i = self.user_to_id[users[i]]
                    user_j = self.user_to_id[users[j]]
                    if rng.random() < 0.5:
                        athlete_a_ids.append(user_i)
                        athlete_b_ids.append(user_j)
                        targets.append(float(score_i > score_j))
                    else:
                        athlete_a_ids.append(user_j)
                        athlete_b_ids.append(user_i)
                        targets.append(float(score_j > score_i))
                    task_ids.append(task_id)
        self.athlete_a_ids = torch.tensor(athlete_a_ids, dtype=torch.long)
        self.athlete_b_ids = torch.tensor(athlete_b_ids, dtype=torch.long)
        self.task_ids = torch.tensor(task_ids, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        if len(self.targets) == 0:
            raise ValueError("Не удалось создать ни одной пары участников")
    def num_usrs(self):
        return len(self.user_to_id)
    def num_tasks(self):
        return len(self.task_to_id)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index):
        return {
            "athlete_a": self.athlete_a_ids[index],
            "athlete_b": self.athlete_b_ids[index],
            "task_id": self.task_ids[index],
            "target": self.targets[index]
        }