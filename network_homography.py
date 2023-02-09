import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random,cv2
import pdb
import kornia,time
from numpy.linalg import inv
import math
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Prediction(nn.Module):
    def __init__(self, num_channels):
        super(Prediction, self).__init__()
        self.num_layers = 4
        self.in_channel = num_channels
        self.kernel_size = 9
        self.num_filters = 64
        self.layer_in = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                  kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_in.weight.data)  
        self.lam_in = nn.Parameter(torch.Tensor([0.01]))

        self.lam_i = []
        self.layer_down = []
        self.layer_up = []
        for i in range(self.num_layers):
            down_conv = 'down_conv_{}'.format(i)  #
            up_conv = 'up_conv_{}'.format(i)
            lam_id = 'lam_{}'.format(i)
            layer_2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,
                                kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_2.weight.data)
            setattr(self, down_conv, layer_2)
            self.layer_down.append(getattr(self, down_conv))
            layer_3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_3.weight.data)
            setattr(self, up_conv, layer_3)
            self.layer_up.append(getattr(self, up_conv))

            lam_ = nn.Parameter(torch.Tensor([0.01]))
            setattr(self, lam_id, lam_)
            self.lam_i.append(getattr(self, lam_id))

    def forward(self, mod):
        p1 = self.layer_in(mod)
        tensor = torch.mul(torch.sign(p1), F.relu(torch.abs(p1) - self.lam_in))  

        for i in range(self.num_layers):
            p3 = self.layer_down[i](tensor)
            p4 = self.layer_up[i](p3)
            p5 = tensor - p4
            p6 = torch.add(p1, p5)
            tensor = torch.mul(torch.sign(p6), F.relu(torch.abs(p6) - self.lam_i[i]))
        return tensor

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Block(nn.Module):
    def __init__(self, inchannels, outchannels, batch_norm=False, pool=True):
        super(Block, self).__init__()
        layers = []
        layers.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class InMIRNet(nn.Module):
    def __init__(self,batch_norm=False,patch_size=256):
        super(InMIRNet, self).__init__()
        self.channel = 1
        self.num_filters = 64
        self.kernel_size = 9
        self.net_u = Prediction(num_channels=self.channel)
        self.conv_u = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_u.weight.data)
        self.conv_u1 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_u1.weight.data)
        self.net_v = Prediction(num_channels=self.channel)
        self.conv_v = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_v.weight.data)
        self.conv_v1 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_v1.weight.data)
        self.net_ck = Prediction(num_channels=self.channel)
        self.conv_x_ck = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel,
                                   kernel_size=self.kernel_size,
                                   stride=1, padding=4, bias=False)
        self.conv_y_ck = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel,
                                   kernel_size=self.kernel_size,
                                   stride=1, padding=4, bias=False)
        self.patch_size = patch_size
        self.basicBlock1 = BasicBlock(1,16)
        self.basicBlock2 = BasicBlock(1,16)
        self.cnn1 = nn.Sequential(
            Block(32, 64, batch_norm),
            Block(64, 64, batch_norm),
            Block(64, 128, batch_norm),
            Block(128, 128, batch_norm, pool=False),
        )
        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * (patch_size//8) * (patch_size//8), 1024), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4 * 2),
        )

    def forward(self, x, y):
        u = self.net_u(x)
        v = self.net_v(y)
        p_x = x - self.conv_u(u)
        p_y = y - self.conv_v(v)
        p_x_att = self.basicBlock1(p_x)
        p_y_att = self.basicBlock2(p_y)
        delta = self.cnn1(torch.cat((p_x_att, p_y_att), dim = 1))
        delta = self.fc(delta)
        delta = delta.view(-1,4,2)

        corners = torch.tensor(
            [
                [0,0],
                [0 + 256, 0],
                [0 + 256, 0 + 256],
                [0, 0 + 256],
            ], dtype=torch.float
        )
        corners = corners.unsqueeze(0).float().cuda()
        corners_hat = corners + delta
        h = kornia.get_perspective_transform(corners, corners_hat)
        patch_b_hat = kornia.warp_perspective(p_x, h, (256,256))
        c_k = self.net_ck(p_x)
        x_c = self.conv_x_ck(c_k)
        x_hat = x_c + self.conv_u(u)
        warp_y_c = self.conv_y_ck(c_k)
        y_c = kornia.warp_perspective(warp_y_c, h, (256,256))
        x_warp = kornia.warp_perspective(x, h, (256,256))
        p_x_warp = kornia.warp_perspective(p_x, h, (256,256))
        y_hat = self.conv_v(v) + y_c

        return x_hat,y_hat,warp_y_c,x_c,x_warp,y_c,p_x,p_y,patch_b_hat,p_x_warp,delta,h
