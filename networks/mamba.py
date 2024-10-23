import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat
from pscan import Scan as pscan


SIZE = 2048
d_state = 16
d_conv = 4
num_states = SIZE
num_observations = 2


class mamba_block(nn.Module):
    def __init__(self, window):
        super(mamba_block, self).__init__()

        self.window = window
        self.in_proj = nn.Linear(SIZE, SIZE * 2)
        self.out_proj = nn.Linear(SIZE, SIZE)
        self.x_proj = nn.Linear(SIZE, SIZE + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(SIZE, SIZE, bias=True)

        self.conv = nn.Conv1d(SIZE, SIZE, kernel_size=d_conv, padding=d_conv - 1)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=SIZE)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(SIZE))

        self.conv = nn.Conv1d(SIZE, SIZE, kernel_size=d_conv, padding=d_conv - 1)

    def forward(self, x, sim):

        x = x.unsqueeze(0)
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[SIZE, SIZE], dim=-1)

        x = nn.BatchNorm1d(x)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)

        x_mask = torch.matmul(x, sim)

        x_1d = x_mask.view(-1, 1, 2048)
        padding = (self.window - 1) // 2
        conv = F.conv1d(x_1d, torch.ones(1, 1, self.window, dtype=x_mask.dtype, device=x_mask.device), padding=padding)
        conv = conv.view(-1, 2048)
        conv = conv / self.window

        x_mask_1 = self.ssm(x_mask)
        x_mask_2 = self.ssm(torch.flip(x_mask, dims=[1]))
        x_mask_3 = self.ssm(conv)
        x_mask_4 = self.ssm(torch.flip(conv, dims=[1]))

        x_merge = x_mask_1 + x_mask_2 + x_mask_3 + x_mask_4

        res = nn.LayerNorm(res)
        res = rearrange(res, 'b l d_in -> b d_in l')
        res = self.conv(res)[:, :, :l]
        res = rearrange(res, 'b d_in l -> b l d_in')

        y = x_merge + F.relu(res)

        return y

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[SIZE, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        _, L, _ = u.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (u.unsqueeze(-1))

        h = torch.zeros(u.size(0), SIZE, d_state, device=deltaA.device)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * u

        return y
