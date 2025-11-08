import os
import torch
from torch.utils.data import Dataset, DataLoader


def load_tiny_shakespeare(block_size=128):  # 新增block_size参数，默认128
    data_path = r"D:\pycharm\transformer-midterm\datasets\tiny_shakespeare\input.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.read()

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data_ids = torch.tensor([stoi[ch] for ch in data], dtype=torch.long)

    n = len(data_ids)
    train_ids = data_ids[:int(n * 0.015)]  # 训练集取98%数据
    val_ids = data_ids[int(n * 0.98):int(n * 0.98) + 2000]  # 验证集仅取前2000条

    class ShakespeareDataset(Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size

        def __len__(self):
            return len(self.data) - self.block_size

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.block_size]
            y = self.data[idx + 1:idx + 1 + self.block_size]
            return x, y

    train_dataset = ShakespeareDataset(train_ids, block_size)
    val_dataset = ShakespeareDataset(val_ids, block_size)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    return train_loader, val_loader, vocab_size, stoi, itos


def load_iwslt2017():
    raise NotImplementedError("iwslt2017数据集加载功能暂未实现")