# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/20 21:45
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn as nn
import math


def bh(m, gi, go):
    print("Grad Input")
    print(gi)
    print("Grad Output")
    print(go)
    # print(go.size())
    gi_ = []
    for i in range(len(gi)):
        gi_.append(gi[i] * 0.1)
    gi_ = tuple(gi_)

    go_ = []
    for i in range(len(go)):
        go_.append(go[i] * 0.1)
    go_ = tuple(go_)
    print('new gi = ', gi_)
    print('new go = ', go_)
    return gi_


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.bias is None:
            return self._backend.Linear()(input, self.weight)
        else:
            return self._backend.Linear()(input, self.weight, self.bias)


x = Variable(torch.FloatTensor([[1, 2, 3]]), requires_grad=True)
mod = nn.Linear(3, 1, bias=False)
mod.register_backward_hook(bh)  # 在这里给module注册了backward hook

out = mod(x)
out.register_hook(lambda grad: 0.1 * grad)  # 在这里给variable注册了 hook
out.backward()
print(['*'] * 20)
print("x.grad", x.grad)
print(mod.weight.grad)