from torch import nn
from SRR_HGNN.models.layers import HGNN_conv
import torch.nn.functional as F
from mamba import mamba_block
import torch
import numpy as np


class HGNN(nn.Module):
    def __init__(self, in_out, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_out, in_out)
        self.hgc2 = HGNN_conv(in_out, in_out)
        self.hgc3 = HGNN_conv(in_out, in_out)
        self.mamba1 = mamba_block()
        self.mamba2 = mamba_block()
        self.mamba3 = mamba_block()

    def forward(self, x, G, sim):
        # hyper 1
        output1, output2 = self.mamba1(x)
        x_ma = torch.matmul(output1, sim) + output2
        x_hg = F.relu(self.hgc1(x_ma, G))
        x_hg = F.dropout(x_hg, self.dropout)

        input_tensor = x_hg.unsqueeze(0).unsqueeze(0)
        pooling_layer = nn.AvgPool2d((2, 1))
        output_tensor = pooling_layer(input_tensor)
        x_half_1 = output_tensor.squeeze().squeeze()

        input_tensor = G.unsqueeze(0).unsqueeze(0)
        pooling_layer = nn.AvgPool2d(2)
        output_tensor = pooling_layer(input_tensor)
        G_half_1 = output_tensor.squeeze().squeeze()

        # hyper 2
        output1_1, output2_1 = self.mamba2(x_half_1)
        x_half_1_ma = torch.matmul(output1_1, sim) + output2_1
        x1_hg = F.relu(self.hgc2(x_half_1_ma, G_half_1))
        x1_hg = F.dropout(x1_hg, self.dropout)

        input_tensor = x1_hg.unsqueeze(0).unsqueeze(0)
        pooling_layer = nn.AvgPool2d((2, 1))
        output_tensor = pooling_layer(input_tensor)
        x_half_2 = output_tensor.squeeze().squeeze()

        input_tensor = G_half_1.unsqueeze(0).unsqueeze(0)
        pooling_layer = nn.AvgPool2d(2)
        output_tensor = pooling_layer(input_tensor)
        G_half_2 = output_tensor.squeeze().squeeze()

        # hyper 3
        output1_2, output2_2 = self.mamba3(x_half_2)
        x_half_2_ma = torch.matmul(output1_2, sim) + output2_2
        x2_hg = F.relu(self.hgc3(x_half_2_ma, G_half_2))
        x2_hg = F.dropout(x2_hg, self.dropout)

        return x2_hg
