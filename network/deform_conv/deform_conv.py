# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/20 22:16
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import numpy as np

import torch
import torch.nn as nn


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, lr_ratio=1.0):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)

        self.offset_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.offset_conv.weight, 0)  # the offset learning are initialized with zero weights
        self.offset_conv.register_backward_hook(self._set_lr)

        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.lr_ratio = lr_ratio

    def _set_lr(self, module, grad_input, grad_output):
        # print('grad input:', grad_input)
        new_grad_input = []

        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                new_grad_input.append(grad_input[i] * self.lr_ratio)
            else:
                new_grad_input.append(grad_input[i])

        new_grad_input = tuple(new_grad_input)
        # print('new grad input:', new_grad_input)
        return new_grad_input

    def forward(self, x):
        offset = self.offset_conv(x)
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]).type_as(x).long()
        offsets_index.requires_grad = False
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        """
            if q is float, using bilinear interpolate, it has four integer corresponding position.
            The four position is left top, right top, left bottom, right bottom, defined as q_lt, q_rb, q_lb, q_rt
        """
        # (b, h, w, 2N)
        q_lt = p.detach().floor()

        """
            Because the shape of x is N, b, h, w, the pixel position is (y, x)
            *┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄→y
            ┊  .(y, x)   .(y+1, x)
            ┊   
            ┊  .(y, x+1) .(y+1, x+1)
            ┊
            ↓
            x

            For right bottom point, it'x = left top'y + 1, it'y = left top'y + 1
        """
        q_rb = q_lt + 1

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()

        """
            For the left bottom point, it'x is equal to right bottom, it'y is equal to left top
            Therefore, it's y is from q_lt, it's x is from q_rb
        """
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)

        """
            y from q_rb, x from q_lt
            For right top point, it's x is equal t to left top, it's y is equal to right bottom 
        """
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        """
            find p_y <= padding or p_y >= h - 1 - padding, find p_x <= padding or p_x >= x - 1 - padding
            This is to find the points in the area where the pixel value is meaningful.
        """
        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        # print('mask:', mask)

        floor_p = torch.floor(p)
        # print('floor_p = ', floor_p)

        """
           when mask is 1, take floor_p;
           when mask is 0, take original p.
           When thr point in the padding area, interpolation is not meaningful and we can take the nearest
           point which is the most possible to have meaningful value.
        """
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        """
            In the paper, G(q, p) = g(q_x, p_x) * g(q_y, p_y)
            g(a, b) = max(0, 1-|a-b|)
        """
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # print('g_lt size is ', g_lt.size())
        # print('g_lt unsqueeze size:', g_lt.unsqueeze(dim=1).size())

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        """
            In the paper, x(p) = ΣG(p, q) * x(q), G is bilinear kernal
        """
        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        """
            x_offset is kernel_size * kernel_size(N) times x. 
        """
        x_offset = self._reshape_x_offset(x_offset, ks)

        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        """
            In torch 0.4.1 grid_x, grid_y = torch.meshgrid([x, y])
            In torch 1.0   grid_x, grid_y = torch.meshgrid(x, y)
        """
        p_n_x, p_n_y = torch.meshgrid(
            [torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
             torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)])
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        p_n.requires_grad = False
        # print('requires_grad:', p_n.requires_grad)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid([
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride)])
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        p_0.requires_grad = False

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)

        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)

        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


from network.deform_conv.deform_conv import DeformConv2D as DeformConv2D_ori
from time import time

if __name__ == '__main__':
    x = torch.randn(4, 3, 255, 255)

    # p_conv = nn.Conv2d(3, 2 * 3 * 3, kernel_size=3, padding=1, stride=1)
    # conv = nn.Conv2d(3, 64, kernel_size=3, stride=3, bias=False)
    #
    # d_conv1 = DeformConv2D(3, 64)
    # d_conv2 = DeformConv2D_ori(3, 64)
    #
    # offset = p_conv(x)
    #
    # end = time()
    # y1 = conv(d_conv1(x, offset))
    # end = time() - end
    # print('#1 speed = ', end)
    #
    # end = time()
    # y2 = conv(d_conv2(x, offset))
    # end = time() - end
    # print('#2 speed = ', end)

    # mask = (y1 == y2)
    # print(mask)
    # print(torch.max(mask))
    # print(torch.min(mask))

    x = torch.randn(4, 3, 255, 255)
    d_conv = DeformConv2D(3, 64)
    conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    end = time()
    y = d_conv(x)
    end = time() - end
    print('speed = ', end)
    print(y.size())

    end = time()
    y = conv(x)
    end = time() - end
    print('speed = ', end)

    if isinstance(d_conv, nn.Conv2d):
        print('Yes')
    else:
        print('No')




