# person_pair.py
"""
Concatenate the features of position, pair (person A and person B) and union.
"""


from utils.transformer import Transformer
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from vgg_v1 import resnet50
from torch_geometric.nn import GatedGraphConv, GCNConv
from einops import repeat, rearrange
from timm import create_model
import hypergraph_utils as hgut
from HGNN import HGNN

from torchvision.models import resnet101
from mamba import mamba_block

ViT_imagenet = create_model('vit_large_patch16_224', pretrained=False, num_classes=2048)
ViT_imagenet_1 = create_model('vit_large_patch16_224', pretrained=False, num_classes=2048)
ViT_imagenet_2 = create_model('vit_large_patch16_224', pretrained=False, num_classes=2048)
print("---success load pretrain ViT---")
ViT_dict = ViT_imagenet.state_dict()

# please download the large ViT model, and input the path
pretrained_model = torch.load(r'/xxx/jx_vit_large_p16_224-4ee7a4dc.pth')
del pretrained_model['head.weight']
del pretrained_model['head.bias']
ViT_dict.update(pretrained_model)
ViT_imagenet.load_state_dict(ViT_dict)
# 1 PRE
ViT_dict = ViT_imagenet_1.state_dict()
pretrained_model = torch.load(r'/xxx/jx_vit_large_p16_224-4ee7a4dc.pth')
del pretrained_model['head.weight']
del pretrained_model['head.bias']
ViT_dict.update(pretrained_model)
ViT_imagenet_1.load_state_dict(ViT_dict)
# 2 PRE
ViT_dict = ViT_imagenet_2.state_dict()
pretrained_model = torch.load(r'/xxx/jx_vit_large_p16_224-4ee7a4dc.pth')
del pretrained_model['head.weight']
del pretrained_model['head.bias']
ViT_dict.update(pretrained_model)
ViT_imagenet_2.load_state_dict(ViT_dict)

print("------------------------------pretrained vit over-----------------------")


class person_pair(nn.Module):
    def __init__(self, num_classes=6):
        super(person_pair, self).__init__()

        self.pair = ViT_imagenet_1
        self.person_a = ViT_imagenet
        self.person_b = self.person_a
        self.bboxes = nn.Linear(10, 2048)

    # x1 = union, x2 = object1, x3 = object2, x4 = bbox geometric info
    def forward(self, x1, x2, x3, x4):
        x1 = self.pair(x1)
        x2 = self.person_a(x2)
        x3 = self.person_b(x3)
        x4 = self.bboxes(x4)

        return x4, x1, x2, x3


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim = 2048, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


SIZE = 512
dim = 2048
heads = 8
dim_head = 64
dropout = 0.
depth = 2
mlp_dim = 1024


def zoh_discretize(x):
    return torch.floor(x)


class network(nn.Module):
    def __init__(self, num_classes=6):
        super(network, self).__init__()

        self.person_pair = person_pair(num_classes)

        # Multi-Head Self-Attention module
        self.cls_token = nn.Parameter(torch.randn(1, 1, 2048))
        self.multiattn_intra = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.multiattn_pairfuse = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.multiattn_inter = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.full_im_net = resnet50(pretrained=False)
        self.obj_unary = nn.Linear(1000, SIZE * 4)
        self.cls_to_size = nn.Linear(2048, num_classes)
        self.ReLU = nn.ReLU(True)
        self.Dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

        self.gcn = GCNConv(4, 2048)
        self.SG_FE = ViT_imagenet_2
        self.SG_TRM = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.SG_vis_global = nn.Parameter(torch.randn(1, 1, 2048))

        self.HGNN1 = HGNN(2048, 0.5)
        self.HGNN2 = HGNN(2048, 0.5)

    def forward(self, union, b1, b2, b_geometric, full_im, img_rel_num, edge_index, scene_graph_bbox, scene_box_num,
                scene_im):
        rois_feature = self.full_im_net(full_im)  # full image =====scene feature
        x4, x1, x2, x3 = self.person_pair(union, b1, b2, b_geometric)  # crop image =====people feature

        full_im_feature = self.obj_unary(rois_feature)
        SG_feature = self.SG_FE(scene_im)
        # scene graph bbox
        count = 0
        scene_graph_pos_info = torch.zeros([0, 2048], dtype=torch.float32).cuda()
        scene_graph_vis_info = torch.zeros([0, 2048], dtype=torch.float32).cuda()
        for i in scene_box_num:
            # vis
            scene_graph_vis_i = SG_feature[count:count + i].cuda()
            scene_graph_vis_i = scene_graph_vis_i[np.newaxis, :, :]
            vis_global_cat = torch.cat((self.SG_vis_global, scene_graph_vis_i), dim=1)
            vis_global_cat = self.SG_TRM(vis_global_cat)[:, 0, :]
            scene_graph_vis_info = torch.cat([scene_graph_vis_info, vis_global_cat], dim=0)

            # pos
            scene_graph_bbox_i = scene_graph_bbox[count:count + i].cuda()
            edge_matrix = torch.ones([i, i]).cuda()
            edge = edge_matrix.nonzero()
            edge = rearrange(edge, 'l p -> p l')
            scene_graph_bbox_i = self.gcn(scene_graph_bbox_i, edge)[0, :].unsqueeze(0)
            scene_graph_pos_info = torch.cat([scene_graph_pos_info, scene_graph_bbox_i], dim=0)

        scene_features = torch.cat((scene_graph_vis_info, SG_feature, full_im_feature, scene_graph_pos_info), 0)
        distances_scene = torch.cdist(
            torch.stack([zoh_discretize(scene_graph_vis_info.T), zoh_discretize(full_im_feature.T),
                         zoh_discretize(scene_graph_pos_info.T)],
                        dim=0),
            torch.stack([zoh_discretize(scene_graph_vis_info.T), zoh_discretize(full_im_feature.T),
                         zoh_discretize(scene_graph_pos_info.T)],
                        dim=0))

        distances_scene = torch.sum(distances_scene, dim=0, keepdim=True)
        max_distance_scene = torch.max(distances_scene)
        normalized_distances_scene = distances_scene / max_distance_scene

        threshold = 0.6

        mask = normalized_distances_scene > threshold
        normalized_distances_masked_scene = torch.where(mask.cuda(), torch.tensor(0.0).cuda(),
                                                        normalized_distances_scene.squeeze(0).cuda())

        # construct feature matrix
        fts_scene = None
        fts_scene = hgut.feature_concat(fts_scene, scene_features)

        # construct hypergraph incidence matrix
        H_scene = None
        tmp_scene = hgut.construct_H_with_KNN(scene_features.cpu().detach().numpy(), K_neigs=[50],
                                              split_diff_scale=False,
                                              is_probH=True, m_prob=1)
        H_scene = hgut.hyperedge_concat(H_scene, tmp_scene)
        G_scene = hgut.generate_G_from_H(H_scene)
        fts_scene = fts_scene.cpu()
        G_scene = torch.Tensor(G_scene).cpu()
        hyperGraph_output_scene = self.HGNN1(fts_scene.cuda(), G_scene.cuda(),
                                             normalized_distances_masked_scene.squeeze(0).cuda())


        person_features = torch.cat((x4, x1, x2, x3), 0)

        distances = torch.cdist(
            torch.stack([zoh_discretize(x4.T), zoh_discretize(x1.T), zoh_discretize(x2.T), zoh_discretize(x3.T)],
                        dim=0),
            torch.stack([zoh_discretize(x4.T), zoh_discretize(x1.T), zoh_discretize(x2.T), zoh_discretize(x3.T)],
                        dim=0))


        distances = torch.sum(distances, dim=0, keepdim=True)
        max_distance = torch.max(distances)
        normalized_distances = distances / max_distance

        mask = normalized_distances > threshold
        count_above_threshold = torch.sum(mask).item()
        normalized_distances_masked = torch.where(mask.cuda(), torch.tensor(0.0).cuda(),
                                                  normalized_distances).squeeze(0)

        # construct feature matrix
        fts_person = None
        fts_person = hgut.feature_concat(fts_person, person_features)

        # construct hypergraph incidence matrix
        H_person = None
        tmp_person = hgut.construct_H_with_KNN(person_features.cpu().detach().numpy(), K_neigs=[50],
                                               split_diff_scale=False,
                                               is_probH=True, m_prob=1)
        H_person = hgut.hyperedge_concat(H_person, tmp_person)
        G_person = hgut.generate_G_from_H(H_person)
        fts_person = fts_person
        G_person = torch.Tensor(G_person)
        hyperGraph_output_person = self.HGNN2(fts_person.cuda(), G_person.cuda(), normalized_distances_masked)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=x2.shape[0])


        x2 = x2[:, np.newaxis, :]
        x3 = x3[:, np.newaxis, :]
        hyperGraph_output_person = hyperGraph_output_person[:, np.newaxis, :]
        fea_attn = torch.cat((cls_tokens, hyperGraph_output_person, x2, x3), dim=-2)
        fea_mhsa = self.multiattn_intra(fea_attn)

        x1 = x1[:, np.newaxis, :]
        x4 = x4[:, np.newaxis, :]
        fea_mhsa = fea_mhsa[:, 0, :].unsqueeze(1)
        hyperGraph_output_scene = hyperGraph_output_scene[:, np.newaxis, :]

        if x1.shape[0] < hyperGraph_output_scene.shape[0]:
            hyperGraph_output_scene = hyperGraph_output_scene[:x1.shape[0], :, :]
        elif x1.shape[0] > hyperGraph_output_scene.shape[0]:
            additional_rows = x1.shape[0] - hyperGraph_output_scene.shape[0]
            additional_info = torch.zeros(additional_rows, 1, 2048)
            hyperGraph_output_scene = torch.cat((hyperGraph_output_scene, additional_info.cuda()), dim=0)

        fea_mhsa = torch.cat((fea_mhsa, hyperGraph_output_scene, x1, x4), dim=-2)
        fea_mhsa = self.multiattn_pairfuse(fea_mhsa)

        cls_attn = fea_mhsa[:, 0, :]

        rel_num_1 = img_rel_num[0]
        count = int(rel_num_1)
        count_img = 0

        rois_feature_1 = scene_graph_vis_info[0].unsqueeze(0).unsqueeze(0)
        img_inter_1 = cls_attn[0:rel_num_1].unsqueeze(0)
        img_inter_1 = torch.cat((rois_feature_1, img_inter_1), dim=1)
        output = self.multiattn_inter(img_inter_1).squeeze(0)
        output = output[1:, :]

        for rel_num in img_rel_num[1:]:
            if rel_num == 1:
                test_cls = cls_attn[count].unsqueeze(dim=0).unsqueeze(dim=0)
                rois_feature_1 = scene_graph_vis_info[count_img].unsqueeze(0).unsqueeze(0)
                img_inter = torch.cat((rois_feature_1, test_cls), dim=1)
                output_1 = self.multiattn_inter(img_inter).squeeze(0)
                output = torch.cat((output, output_1[1:, :]), dim=0)
            else:
                img_inter = cls_attn[count:(count + rel_num)].unsqueeze(0)
                rois_feature_1 = scene_graph_vis_info[count_img].unsqueeze(0).unsqueeze(0)
                img_inter = torch.cat((rois_feature_1, img_inter), dim=1)
                output_1 = self.multiattn_inter(img_inter).squeeze(0)
                output = torch.cat((output, output_1[1:, :]), dim=0)
            count += rel_num
            count_img = count_img + 1

        result = self.cls_to_size(output)

        return result

