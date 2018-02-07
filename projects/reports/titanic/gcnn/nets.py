import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from skorch import NeuralNet
from skorch.utils import to_numpy
from sklearn.metrics import log_loss

from gcnn.layers import GraphChebyConv, GraphFourierConv


class BaselineCNN(nn.Module):
    def __init__(self, conv1_dim=16, conv2_dim=32):
        super().__init__()

        self.c1 = nn.Sequential(
            # groups 2 performe two parallel pairs of convolution
            nn.Conv2d(2, conv1_dim, kernel_size=7, stride=1, padding=2, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(conv1_dim, conv2_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(conv2_dim * 16 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, x2):
        out = self.c1(x)
        out = self.c2(out)
        out = out.view(out.size(0), -1)
        out = torch.cat([out, x2], 1)
        out = self.fc(out)
        return out
    
    
class BaselineCNNSimple(nn.Module):
    def __init__(self, conv1_dim=16, conv2_dim=32):
        super().__init__()

        self.c1 = nn.Sequential(
            # groups 2 performe two parallel pairs of convolution
            nn.Conv2d(2, conv1_dim, kernel_size=7, stride=1, padding=2, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(conv1_dim, conv2_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(conv2_dim * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    
class GraphCNNSimple(nn.Module):
    
    def __init__(self, lf0, lf2, conv1_dim=16, conv2_dim=32, k=None):
        super().__init__()
        
        if k:
            conv_1 = GraphChebyConv(lf0, 1, conv1_dim, k, bias=True)
            conv_2 = GraphChebyConv(lf2, conv1_dim, conv2_dim, k, bias=True)
        else:
            conv_1 = GraphFourierConv(lf0, 1, conv1_dim, bias=True)
            conv_2 = GraphFourierConv(lf2, conv1_dim, conv2_dim, bias=True)

        self.gc1 = nn.Sequential(
            conv_1,
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.gc2 = nn.Sequential(
            conv_2,
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.fc = nn.Sequential(
            nn.Linear(lf2.size(0) // 4 * conv2_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out = self.gc1(x)
        out = self.gc2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    
class GraphCNN(nn.Module):
    
    def __init__(self, lf0, lf2, conv1_dim=16, conv2_dim=32, k=None):
        super().__init__()
        
        if k:
            conv_1 = GraphChebyConv(lf0, 1, conv1_dim, k, bias=True)
            conv_2 = GraphChebyConv(lf2, conv1_dim, conv2_dim, k, bias=True)
        else:
            conv_1 = GraphFourierConv(lf0, 1, conv1_dim, bias=True)
            conv_2 = GraphFourierConv(lf2, conv1_dim, conv2_dim, bias=True)

        self.gc1 = nn.Sequential(
            conv_1,
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.gc2 = nn.Sequential(
            conv_2,
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.fc = nn.Sequential(
            nn.Linear(lf2.size(0) // 4 * conv2_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, x2):
        out = self.gc1(x)
        out = self.gc2(out)
        out = out.view(out.size(0), -1)
        out = torch.cat([out, x2], 1)
        out = self.fc(out)
        return out

    
class PaperSimpleGC(nn.Module):
    """
    GC10
    """
    
    def __init__(self, l0):
        super().__init__()

        self.gc1 = nn.Sequential(
            # GraphFourierConv(f0.cuda() if cuda else f, 1, 10, bias=True),
            GraphChebyConv(l0, 1, 10, 20, bias=True),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(len(f0) * 10, 10),
            nn.ReLU(),
            nn.Softmax(1)
        )

    def forward(self, x):
        out = self.gc1(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    
class PaperGCFC(nn.Module):
    """
    GC32-P4-GC64-P4-FC512
    """
    
    def __init__(self, l0, l2, conv1_dim=32, conv2_dim=64):
        super().__init__()

        self.gc1 = nn.Sequential(
            # GraphFourierConv(l0.cuda() if cuda else l0, 1, conv1_dim, bias=True),
            GraphChebyConv(l0, 1, conv1_dim, 25, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.gc2 = nn.Sequential(
            # GraphFourierConv(l2.cuda() if cuda else l2, conv1_dim, conv2_dim, bias=True),
            GraphChebyConv(l2, conv1_dim, conv2_dim, 25, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.fc = nn.Sequential(
            nn.Linear(l2.size(0) // 4 * conv2_dim, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Softmax(1)
        )

    def forward(self, x):
        out = self.gc1(x)
        out = self.gc2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    