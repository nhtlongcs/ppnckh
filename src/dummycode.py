import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
from losses import DiceLoss
from models import ResidUNet

if __name__ == "__main__":
    dev = torch.device('cpu')
    net = ResidUNet(3, 2).to(dev)
    print(net)
    criterion = DiceLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    for iter_id in range(100):
        inps = torch.rand(5, 3, 100, 100).to(dev)
        lbls = torch.randint(low=0, high=2, size=(5, 100, 100)).to(dev)
        outs = net(inps)
        print(outs.shape, lbls.shape)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()
        print(iter_id, loss.item())
