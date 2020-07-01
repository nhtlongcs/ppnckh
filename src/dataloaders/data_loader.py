import torch

def data_loader(dataset, batch_size, shuffle=False):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
