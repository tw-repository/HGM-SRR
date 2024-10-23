from torch import nn
from layers import HGNN_conv
import torch.nn.functional as F
from mamba import mamba_block
import torch
import hypergraph_utils as hgut
import numpy as np


class HGNN(nn.Module):
    def __init__(self, in_out, dropout=0.5, window=2):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_out, in_out)
        self.hgc2 = HGNN_conv(in_out, in_out)
        self.hgc3 = HGNN_conv(in_out, in_out)
        self.mamba1 = mamba_block(window)
        self.mamba2 = mamba_block(window)
        self.mamba3 = mamba_block(window)

    def forward(self, x, G, sim):
        # hyper 1
        x_ma = self.mamba1(x, sim)
        x_hg = F.relu(self.hgc1(x_ma, G))
        x_hg = F.dropout(x_hg, self.dropout)

        input_tensor = x_hg.unsqueeze(0).unsqueeze(0)
        pooling_layer = nn.AvgPool2d((2, 1))
        output_tensor = pooling_layer(input_tensor)
        x_half_1 = output_tensor.squeeze().squeeze()

        H_1 = None
        tmp_1 = hgut.construct_H_with_KNN(x_half_1.cpu().detach().numpy(), K_neigs=[10],
                                          split_diff_scale=False,
                                          is_probH=True, m_prob=1)
        H_1 = hgut.hyperedge_concat(H_1, tmp_1)
        G_1 = hgut.generate_G_from_H(H_1)
        G_1 = torch.Tensor(G_1)
        input_tensor = G_1.unsqueeze(0).unsqueeze(0)
        pooling_layer = nn.AvgPool2d(2)
        output_tensor = pooling_layer(input_tensor)
        G_half_1 = output_tensor.squeeze().squeeze()

        # hyper 2
        x_half_1_ma = self.mamba2(x_half_1, sim)
        x1_hg = F.relu(self.hgc2(x_half_1_ma, G_half_1))
        x1_hg = F.dropout(x1_hg, self.dropout)

        input_tensor = x1_hg.unsqueeze(0).unsqueeze(0)
        pooling_layer = nn.AvgPool2d((2, 1))
        output_tensor = pooling_layer(input_tensor)
        x_half_2 = output_tensor.squeeze().squeeze()

        H_2 = None
        tmp_2 = hgut.construct_H_with_KNN(x_half_2.cpu().detach().numpy(), K_neigs=[10],
                                          split_diff_scale=False,
                                          is_probH=True, m_prob=1)
        H_2 = hgut.hyperedge_concat(H_2, tmp_2)
        G_2 = hgut.generate_G_from_H(H_2)
        G_2 = torch.Tensor(G_2)
        input_tensor = G_2.unsqueeze(0).unsqueeze(0)
        pooling_layer = nn.AvgPool2d(2)
        output_tensor = pooling_layer(input_tensor)
        G_half_2 = output_tensor.squeeze().squeeze()

        # hyper 3
        x_half_2_ma = self.mamba3(x_half_2, sim)
        x2_hg = F.relu(self.hgc3(x_half_2_ma, G_half_2))
        x2_hg = F.dropout(x2_hg, self.dropout)

        return x2_hg
