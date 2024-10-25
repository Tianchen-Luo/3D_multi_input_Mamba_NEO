import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
 
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
 
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
 
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, mamba_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.mamba_layer = mamba_layer
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.mamba_layer is not None:
            global_att = self.mamba_layer(out)   
            out += global_att

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_res_layer(inplanes, planes, blocks, stride=1, mamba_layer=None):
 
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes, stride),
            nn.BatchNorm3d(planes),
        )

    layers = [BasicBlock(inplanes, planes, stride, downsample, mamba_layer)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes, mamba_layer=mamba_layer))

    return nn.Sequential(*layers)


class MambaLayer(nn.Module):
 
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.nin = conv1x1(dim, dim)
        self.norm = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_mamba = self.mamba(x_flat)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out += x  

        return out



    

class Multi_input_MambaClassifier(nn.Module):

    def __init__(self, in_ch=1, channels=64, blocks=4, num_classes=1, num_inputs=7):
        super(Multi_input_MambaClassifier, self).__init__()

        self.num_inputs = num_inputs
        # First input's independent layers
        self.in_conv_0 = nn.Sequential(nn.Conv3d(in_ch, channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm3d(channels), nn.ReLU(inplace=True))
        self.layer_0 = make_res_layer(channels, channels * 2, blocks, stride=2, mamba_layer=MambaLayer(channels * 2))
        self.avgpool_0 = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Shared layers for the remaining inputs
        self.in_conv_shared = nn.Sequential(nn.Conv3d(in_ch, channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm3d(channels), nn.ReLU(inplace=True))
        self.layer_shared = make_res_layer(channels, channels * 2, blocks, stride=2, mamba_layer=MambaLayer(channels * 2))
        self.avgpool_shared = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(channels * 2 * num_inputs, num_classes)

    def forward(self, x):

        batch_size, num_inputs, D, H, W = x.shape
        outputs = []

        # Process the first input with its independent layers
        x0 = x[:, 0, :, :, :].unsqueeze(1)   
        x0 = self.in_conv_0(x0)
        x0 = self.layer_0(x0)
        x0 = self.avgpool_0(x0)
        x0 = torch.flatten(x0, 1)
        outputs.append(x0)

        # Process the remaining inputs with shared layers
        for i in range(1, num_inputs):
            xi = x[:, i, :, :, :].unsqueeze(1)  
            xi = self.in_conv_shared(xi)
            xi = self.layer_shared(xi)
            xi = self.avgpool_shared(xi)
            xi = torch.flatten(xi, 1)
            outputs.append(xi)

         
        x = torch.cat(outputs, dim=1)
        x = self.fc(x)
        return x