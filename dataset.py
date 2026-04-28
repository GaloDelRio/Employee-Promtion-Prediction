import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, x_cat: torch.Tensor, x_num: torch.Tensor, y: torch.Tensor):
        self.x_cat = x_cat
        self.x_num = x_num
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_num[idx], self.y[idx]
