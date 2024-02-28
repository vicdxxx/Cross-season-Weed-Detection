import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import os
import math
import torch.nn.init as init
import numpy as np

#da_device = torch.device('cuda')

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    reversed_input = GradReverse.apply(x, lambd)
    return reversed_input


class netD_pixel_v0(nn.Module):
    def __init__(self, context=False):
        super(netD_pixel_v0, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.context = context
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            x = self.conv3(x)
            return torch.sigmoid(x), feat
        else:
            x = self.conv3(x)
            return torch.sigmoid(x)


class netD_pixel(nn.Module):
    def __init__(self, context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.context = context
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
        normal_init(self.conv1, 0, 0.1)
        normal_init(self.conv2, 0, 0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return torch.sigmoid(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class netD_v0(nn.Module):
    def __init__(self, context=False):
        super(netD_v0, self).__init__()
        self.conv1 = conv3x3(256, 256, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv3x3(256, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        if self.context:
            feat = x
        x = self.fc(x)
        if self.context:
            return x, feat
        else:
            return x

class netD(nn.Module):
    def __init__(self, context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(256, 128, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = conv3x3(128, 32, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32, 2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 32)
        if self.context:
            feat = x
        x = self.fc(x)
        if self.context:
            return x, feat
        else:
            return x

class net_gcr(nn.Module):
    def __init__(self, out_channels):
        super(net_gcr, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.conv2 = conv3x3(512, 256, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv3 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                # not a perfect approximation
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.avg_pool(x)
        x = self.conv3(x).squeeze(-1).squeeze(-1)
        return x

class net_gcr_simple(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(net_gcr_simple, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x).squeeze(-1).squeeze(-1)
        return x


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)  # 2
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]
        # for pass amp
        self.weight = nn.Parameter(torch.Tensor(output_dim))

    def forward(self, input_list):
        return_list = []
        for i in range(self.input_num):
            self.random_matrix[i] = self.random_matrix[i].to(input_list[i].device)
            return_list += [torch.mm(input_list[i], self.random_matrix[i])]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def get_fea_list(fea, pre, num_classes):
    fea = fea.view(-1, fea.size(-1))
    pre = pre.view(-1, pre.size(-1))
    _, max_ind = torch.max(pre, -1)
    max_ind %= num_classes
    fea_list = []
    # no bg class
    try:
        for i in range(0, num_classes):
            max_ind_i = (max_ind == i)
            if fea[max_ind_i].numel():
                fea_list.append(fea[max_ind_i].mean(0))
            else:
                fea_list.append(torch.tensor([]))
    except Exception as e:
        print(e)
        for i in range(0, num_classes):
            fea_list.append(torch.tensor([]))
    return fea_list

def get_feature_vector(f1, f2, softmax, RandomLayer, num_classes):
    bs = f1.size(0)
    f1 = f1.permute(0, 2, 3, 1).contiguous()
    f1 = f1.view(-1, f1.size(-1))
    f2 = f2.view(-1, f2.size(-1))
    f2 = f2.view(f2.size(0), -1, num_classes)
    f2 = softmax(f2).view(f2.size(0), -1)
    feat = RandomLayer([f1, f2])
    feat = feat.pow(2).mean(1)
    feat = feat.view(bs, -1)
    feat = F.normalize(feat, p=2, dim=1).mean(0)
    return feat

def Moving_average(cur_fea_lists, fea_lists):
    for i in range(len(cur_fea_lists)):
        for j in range(len(cur_fea_lists[0])):
            if cur_fea_lists[i][j].numel():
                if fea_lists[i][j].numel():
                    fea_lists[i][j] = fea_lists[i][j].detach()*0.7 + cur_fea_lists[i][j]*0.3
                else:
                    fea_lists[i][j] = cur_fea_lists[i][j]
            else:
                fea_lists[i][j] = fea_lists[i][j].detach()
    return fea_lists


def get_kl_loss(pre_lists, softmax, num_classes):
    kl_lists = []
    for pre in pre_lists:
        pre = pre.view(-1, pre.size(-1))/2
        pre = pre.view(pre.size(0), -1, num_classes)
        pre = softmax(pre).mean(1)
        _, max_ind = torch.max(pre, -1)
        kl_list = []
        # no bg class
        for i in range(0, num_classes):
            max_ind_i = (max_ind == i)
            if pre[max_ind_i].numel():
                kl_list.append(pre[max_ind_i].mean(0)+1e-6)
            else:
                kl_list.append(torch.tensor([]))
        kl_lists.append(kl_list)
    loss = torch.tensor(0).float().cuda()
    cnt = 0
    p_list, q_list = kl_lists
    for i in range(num_classes):
        p, q = p_list[i], q_list[i]
        if p.numel() and q.numel():
            tmp = p*torch.log(p/q) + q*torch.log(q/p)
            loss += tmp.mean()/2
            cnt += 1
    if cnt:
        loss /= cnt
    return loss

def dcbr_w1_weight(cls_pre):
    # cls_pre.shape = [bs, num_classes]
    ind = (cls_pre > 0.5).float()
    cls_pre_valid = cls_pre * ind
    w1 = cls_pre_valid.sum(1)/(ind.sum(1)+1e-7) + 1
    return w1


def weight_ce_loss(pre, target, weight):
    # target = 0/1
    # pre.shape = [bs, 2]
    # weight.shape = [bs]
    pre_softmax = F.softmax(pre, dim=1).clamp(1e-6, 1)  # [bs, 2]
    #
    pre_softmax = pre_softmax[:, target]  # [bs]
    loss = -weight*torch.log(pre_softmax)
    return loss.mean()


def gt_classes2cls_onehot(batch_size, cls, batch_idx, classed_num=12):
    # revised
    batch_idx_ = np.array(batch_idx)
    im_idx = np.unique(batch_idx)
    bs = batch_size
    cls_onehot = np.zeros((bs, classed_num), np.float32)
    for i in range(bs):
        target = cls[batch_idx_ == i]
        for one_target in target:
            if one_target == 0:
                continue
            cls_onehot[i, int(one_target.cpu().numpy())-1] = 1
    return cls_onehot



def get_pa_losses(fea_lists, fea_lists_t):
    # compute intra and inter loss
    loss = 0
    for i in range(len(fea_lists)):
        fea_list = fea_lists[i]
        fea_list_t = fea_lists_t[i]
        loss += get_pa_loss(fea_list, fea_list_t)
    return loss


def get_pa_loss(fea_list, fea_list_t):
    # compute intra and inter loss

    # intra loss
    intra_loss = 0
    cnt = 0
    for (fea, fea_t) in zip(fea_list, fea_list_t):
        if fea.numel() and fea_t.numel():
            intra_loss += torch.pow(fea-fea_t, 2.0).mean()
            cnt += 1
    if cnt:
        intra_loss /= cnt

    # inter loss
    inter_loss = 0
    cnt = 0
    cls_num = len(fea_list)  # 20
    for i in range(cls_num):
        src_1 = fea_list[i]
        tgt_1 = fea_list_t[i]
        for j in range(i+1, cls_num):
            src_2 = fea_list[j]
            tgt_2 = fea_list_t[j]

            if src_1.numel():
                if src_2.numel():
                    inter_loss += contrasive_separation(src_1, src_2)
                    cnt += 1
                if tgt_2.numel():
                    inter_loss += contrasive_separation(src_1, tgt_2)
                    cnt += 1
            if tgt_1.numel():
                if src_2.numel():
                    inter_loss += contrasive_separation(tgt_1, src_2)
                    cnt += 1
                if tgt_2.numel():
                    inter_loss += contrasive_separation(tgt_1, tgt_2)
                    cnt += 1
    if cnt:
        inter_loss /= cnt

    return intra_loss + inter_loss

def contrasive_separation(f1, f2):
    dis = torch.pow(f1-f2, 2.0).mean().sqrt()
    loss = torch.pow(torch.max(1 - dis, torch.tensor(0).float().cuda()), 2.0)
    loss *= torch.pow(1-dis, 2.0)
    return loss
