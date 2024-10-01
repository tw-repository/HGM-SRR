# test.py
"""
test the val acc
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import random

from dataset_HGNN import DataSet

seed = 10
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='PyTorch Relationship')
parser.add_argument('--mode', dest="mode", default="PISC_Fine", type=str,
                    help='PISC_Fine, PISC_Coarse or PIPA')
parser.add_argument('--network', default='GGNN_attn', type=str, help='Network name.')
parser.add_argument('--scene', default=True, type=bool, help='whether scene node exists')
# your model path
parser.add_argument('--model_file', default=r"/xxx/hgnn_model_xxx.pth.tar",
                    type=str, help='Model file to be tested.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (defult: 4)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--scale-size', default=256, type=int, help='input size')
parser.add_argument('--crop-size', default=224, type=int, help='crop size')

args = parser.parse_args()
print(args)
if args.mode == "PISC_Fine":
    data_dir = r"/xxx/inputFile/image"
    num_class = 6
    train_list = r"/xxx/data/inputfile/PISCF_train.pkl"
    test_list = r"/xxx/data/inputfile/PISCF_test.pkl"
    labels = ['Fri', 'Fam', 'Cou', 'Pro', 'Com', 'No']
elif args.mode == "PISC_Coarse":
    data_dir = r"/xxx/inputFile/image"
    num_class = 3
    train_list = r"/xxx/data/inputfile/PISCC_train.pkl"
    test_list = r"/xxx/data/inputfile/PISCC_test.pkl"
    labels = ['Intimate', 'Non-intimate', 'No']
else:
    data_dir = r"/xxx/inputFile/image"
    num_class = 16
    train_list = r"/xxx/data/inputfile/PIPA_train.pkl"
    test_list = r"/xxx/data/inputfile/PIPA_test.pkl"
    labels = ['father-child', 'mother-child', 'grandpa-grandchild', 'grandma-grandchild', 'friends', 'siblings',
              'classmates', 'lovers/spouses', 'presenter-audience', 'teacher-student', 'trainer-trainee',
              'leader-subordinate', 'band members', 'dance team members', 'sport team members', 'colleagues']

def vg_collate(data):
    Name = []
    Union = []
    Obj1 = []
    Obj2 = []
    Bpos = []
    Target = []
    Full_im = []
    Img_rel_num = []
    scene_box = []
    scene_box_num = []
    scene_im = []

    for d in data:
        name, union, obj1, obj2, bpos, target, full_im, img_rel_num, \
        scene_bbox_tensor, scene_bbox_num, scene_bbox_crop_all = d

        Name.append(name)
        Union.append(union)
        Obj1.append(obj1)
        Obj2.append(obj2)
        Bpos.append(bpos)
        Target.append(target)
        Full_im.append(full_im)
        Img_rel_num.append(img_rel_num)
        scene_box.append(scene_bbox_tensor)
        scene_box_num.append(scene_bbox_num)
        scene_im.append(scene_bbox_crop_all)

    Union = torch.cat(Union, 0)
    Obj1 = torch.cat(Obj1, 0)
    Obj2 = torch.cat(Obj2, 0)
    Bpos = torch.cat(Bpos, 0)
    Target = torch.cat(Target, 0)
    Full_im = torch.cat(Full_im, 0)
    Full_im = Full_im.view(-1, 3, 224, 224)
    Img_rel_num = torch.cat(Img_rel_num, 0)
    scene_bbox = torch.cat(scene_box, 0)
    scene_box_num = torch.cat(scene_box_num, 0)
    scene_im = torch.cat(scene_im, 0)

    return Name, Union, Obj1, Obj2, Bpos, Target, Full_im, Img_rel_num, scene_bbox, scene_box_num, scene_im


def get_test_set(data_dir, test_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_size = args.scale_size
    crop_size = args.crop_size

    test_data_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize])  # what about horizontal flip

    test_full_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])  # what about horizontal flip

    test_set = DataSet(data_dir, test_list, test_data_transform, test_full_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers, collate_fn=vg_collate,
                             batch_size=args.batch_size, shuffle=False)
    return test_loader

def generate_graph(rel_num, scene=args.scene):
    if scene:
        numNode = rel_num + 1
    else: numNode = rel_num
    edge_index = []
    if numNode != 1:
        for i in range(numNode):
            for j in range(numNode):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.t().contiguous()
    else:
        edge_index = [[], []]
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.contiguous()
    return edge_index


def validate(val_loader, model):
    model.eval()
    tp, p, r = {}, {}, {}
    for batch_data in tqdm(val_loader):

        name, union, obj1, obj2, bpos, target, full_im, img_rel_num, \
        scene_graph_bbox, scene_box_num, scene_im = batch_data

        target = target.cuda()
        union_var = torch.autograd.Variable(union).cuda()
        obj1_var = torch.autograd.Variable(obj1).cuda()
        obj2_var = torch.autograd.Variable(obj2).cuda()
        bpos_var = torch.autograd.Variable(bpos).cuda()
        full_im_var = torch.autograd.Variable(full_im).cuda()
        scene_graph_bbox = torch.autograd.Variable(scene_graph_bbox).cuda()
        scene_box_num = torch.autograd.Variable(scene_box_num).cuda()
        scene_im = torch.autograd.Variable(scene_im).cuda()

        edge_index = generate_graph(img_rel_num[0])

        if args.scene:
            count = img_rel_num[0] + 1
        else:
            count = img_rel_num[0]
        for rel_num in img_rel_num[1:]:
            edge_index = torch.cat((edge_index, generate_graph(rel_num) + count), dim=1)
            if args.scene:
                count += rel_num + 1
            else:
                count += rel_num

        img_rel_num = torch.autograd.Variable(img_rel_num).cuda()
        edge_index = torch.autograd.Variable(edge_index).cuda()

        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            output = model(union_var, obj1_var, obj2_var, bpos_var, full_im_var,
                           img_rel_num, edge_index, scene_graph_bbox, scene_box_num, scene_im)

        output_f = F.softmax(output, dim=1)
        output_np = output_f.data.cpu().numpy()
        pre = np.argmax(output_np, 1)
        t = target_var.data.cpu().numpy()

        for i, item in enumerate(t):

            true_i = np.array(item)

            pre_i = np.array(pre[i])

            if item in r:
                r[item] += 1
            else:
                r[item] = 1

            if pre[i] in p:
                p[pre[i]] += 1
            else:
                p[pre[i]] = 1

            if pre[i] == item:
                if item in tp:
                    tp[item] += 1
                else:
                    tp[item] = 1

    precision = {}
    recall = {}
    tp_total = 0

    for k in tp.keys():
        precision[k] = float(tp[k]) / float(p[k])
        recall[k] = (float(tp[k])) / float(r[k])
        tp_total += tp[k]

    p_total = 0

    for k in p.keys():
        p_total += p[k]
    precision_total = float(tp_total) / float(p_total)


    return precision_total, recall


def init_network(net, num_class):
    # Initialize the network.
    if net == 'GGNN_attn':
        from network import network

    model = network(num_class)
    return model


if __name__ == '__main__':
    val_loader = get_test_set(data_dir=data_dir, test_list=test_list)
    model = init_network(args.network, num_class)
    trained_model = torch.load(args.model_file)
    model_dict = model.state_dict()
    trained_model = {k.replace('module.', ''): v for k, v in
                                    trained_model['state_dict'].items()}
    trained_model_dict = {k: v for k, v in trained_model.items() if k in model_dict}
    model_dict.update(trained_model_dict)
    model.load_state_dict(model_dict)

    # torch.backends.cudnn.enabled = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()

    acc, recall = validate(val_loader, model)
    print(recall)
    print("Acc: {:.2%}".format(acc))

