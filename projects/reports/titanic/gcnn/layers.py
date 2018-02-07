import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class GraphFourierConv(nn.Module):
    def __init__(self, fourier_basis, in_channels, out_channels, bias=True):
        super().__init__()

        self.n = fourier_basis.size(0)
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert fourier_basis.size(1) == self.n
        self.u = Variable(fourier_basis, requires_grad=False)  # n x n
        self.ut = self.u.t()

        self.weight = nn.Parameter(torch.Tensor(self.n, self.out_channels, self.in_channels))  # n x out x in

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels, 1))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # batch x in x m

        # fourier
        out = x.view(-1, self.n)  # (batch * in) x m
        out = out @ self.u  # (batch * in) x m

        # filter
        out = out.view(-1, self.in_channels, self.n)  # batch x in x m
        out = out.permute([2, 1, 0])  # m x in x batch
        out = self.weight @ out  # m x out x batch

        # un-fourier
        out = out.transpose(0, 2).contiguous()  # batch x out x m
        out = out.view(-1, self.n)  # (batch * out) x m
        out = out @ self.ut  # (batch * out) x m
        out = out.view(-1, self.out_channels, self.n)  # batch x out x m

        if self.bias is not None:
            out = out + self.bias  # batch x out x m

        return out

    def __repr__(self):
        return '{}(laplacian, {}, {}, bias={})'.format(__class__.__name__, self.in_channels, self.out_channels,
                                                       self.bias.size())



class GraphChebyConv(nn.Module):
    def __init__(self, laplacian, in_channels, out_channels, k, bias=True):
        super().__init__()

        self.n = laplacian.size(0)
        self.k = 25
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert laplacian.size(1) == self.n

        lmax = torch.symeig(laplacian)[0].max()
        assert 0 < lmax <= 2.0001, 'lmax='.format(lmax)
        self.l = Variable(2 / lmax * laplacian - torch.eye(self.n).cuda(), requires_grad=False) # n x n

        self.weight = nn.Parameter(torch.Tensor(self.in_channels * self.k, self.out_channels)) # (in * k) x out

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels, 1))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # batch x in x m

        out = x.view(-1, self.n) # (batch * in) x m
        out = out.t() # m x (batch * in)

        # chebyshev
        xs = [out, self.l @ out]

        # def concat(c, t):
        #    return torch.cat([c, t.unsqueeze(0)])

        # x2 = out
        # x1 = self.l @ out
        # xs = concat(x2.unsqueeze(0), x1)

        # s = timer()
        for k in range(2, self.k):
            #    x0 = 2 * self.l @ x1 - x2
            #    xs = concat(xs, x0)
            #    x1, x2 = x0, x1
            xs.append(2 * self.l @ xs[k - 1] - xs[k - 2])

        xs = torch.stack(xs) # k x m x (batch * in)
        out = xs

        # m = timer()
        # filter
        # out = xs.transpose(0, 2) # (batch * in) x m x k
        # out = out @ self.weight # (batch * in) x m x out
        # out = out.transpose(1, 2).contiguous() # (batch * in) x out x m
        # out = out.view(-1, self.in_channels, self.out_channels, self.n) # batch x in x out x m

        out = out.view(self.k, self.n, x.size(0), self.in_channels) # k x m x batch x in
        out = out.permute([2, 1, 3, 0]).contiguous() # batch x m x in x k
        out = out.view(x.size(0) * self.n, self.in_channels * self.k) # (batch * m) x (in * k)

        out = out @ self.weight # (batch * m) x out
        out = out.view(x.size(0), self.n, self.out_channels) # batch x m x out
        out = out.transpose(1, 2) # batch x out x m

        # e = timer()
        # print((m - s) / 1000, (e - m) / 1000)

        # sum in dim + bias
        # out = out.sum(1) # batch x out x m
        if self.bias is not None:
            out = out + self.bias # batch x out x m

        return out

    def __repr__(self):
        return '{}(fourier_basis, {}, {}, bias={})'.format(__class__.__name__, self.in_channels, self.out_channels, self.bias.size())