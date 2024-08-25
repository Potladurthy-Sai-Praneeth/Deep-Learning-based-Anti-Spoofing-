import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from depth_map_model import FineTuneDepthAnything


class CDC(nn.Module):
    '''
    This class performs central difference convolution (CDC) operation. 
    First the normal convolution is performed and then the convolution is performed with the squeezed version of kernel and the difference is the result.
    Args :
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The size of the kernel.
        stride : int
            The stride of the convolution.
        padding : int
            The padding of the convolution.
        dilation : int
            The dilation of the convolution.
        groups : int
            The number of groups.
        bias : bool
            Whether to use bias or not.
        theta : float
            The value of theta.
    Returns :
        out_normal -  torch.tensor
            The resultant image/channels of the CDC operation.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(CDC, self).__init__()
        self.bias= bias
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.theta = theta
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding if kernel_size==3 else 0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)
        # if conv.weight is (out_channels, in_channels, kernel_size, kernel_size),
        # then the  self.conv.weight.sum(2) will return (out_channels, in_channels,kernel_size)
        # and self.conv.weight.sum(2).sum(2) will return (out_channels,n_channels)
        kernel_diff = self.conv.weight.sum(2).sum(2)
        # Here we are adding extra dimensions such that the kernel_diff is of shape (out_channels, in_channels, 1, 1) so that convolution can be performed.
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.bias, stride=self.stride, padding=0, groups=self.groups)
        return out_normal - self.theta * out_diff
    

# class ClassifierUCDCN(nn.Module):
#     '''
#     A class trained on top of the depth-anything model to classify the spoofing attacks. It is a simple binary classifier.
#     Args :
#         model_pth_path : str
#             The path to the trained depth map model.
#         device : torch.device
#             The device on which the model is loaded.
#         dropout : float
#             The dropout probability.
#     Returns :
#         sigmoid(linear_2) : torch.tensor
#             The output of the classifier.
#         cdc_out : torch.tensor
#             The output of the depth-anything model.
#     '''
#     def __init__(self,model_pth_path,device,dropout=0.5):
#         super(ClassifierUCDCN, self).__init__()
#         self.cdc_net = FineTuneDepthAnything(device).to(device)
#         self.cdc_net.load_state_dict(torch.load(model_pth_path, map_location=device))
#         self.cdc_net.eval()
        
#         self.layers =8
#         self.dropout_prob = dropout
#         self.img_size = (252, 252)
#         self.hidden_size = 100
#         self.conv1 = CDC(3,self.layers,kernel_size=3,stride=1,padding=1)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.AvgPool2d(kernel_size=2,stride=2)
#         self.conv2 = CDC(self.layers,1,kernel_size=3,stride=1,padding=1)
#         # Maxpool
#         self.linear_1 = nn.Linear((self.img_size[0]//4 * self.img_size[1]//4), self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_prob)
#         self.linear_2 = nn.Linear(self.hidden_size, 2)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, inp):
#         with torch.no_grad():
#             cdc_out = self.cdc_net(inp)
#         conv1 = self.conv1(cdc_out.repeat(1,3,1,1))
#         relu1 = self.relu(conv1)
#         maxpool = self.maxpool(relu1)
#         conv2 = self.conv2(maxpool)
#         relu2 = self.relu(conv2)
#         maxpool2 = self.maxpool(relu2)
#         linear_1 = self.linear_1(maxpool2.view(-1, self.img_size[0]//4 * self.img_size[1]//4))
#         dropout = self.dropout(linear_1)
#         linear_2 = self.linear_2(dropout)
#         return self.sigmoid(linear_2),cdc_out

class ClassifierUCDCN(nn.Module):
    '''
        A class trained on top of the depth-anything model to classify the spoofing attacks. It is a simple binary classifier.
        Args :
            depth_map_path : str
                The path to the trained depth map model.
            device : torch.device
                The device on which the model is loaded.
            dropout : float
                The dropout probability.
            load_depth_model : bool
                Flag indicating whether to load the depth model or not.
        Returns :
            sigmoid(linear_2) : torch.tensor
                The output of the classifier.
            cdc_out : torch.tensor
                The output of the depth-anything model.
    '''

    def __init__(self, depth_map_path=None, device='cpu', dropout=0.5, load_depth_model=True):
        super(ClassifierUCDCN, self).__init__()
        self.cdc_net = FineTuneDepthAnything(device, load_trained=load_depth_model, model_path=depth_map_path).to(device)
#         self.cdc_net.load_state_dict(torch.load(model_pth_path, map_location=device))
        self.cdc_net.eval()
        
        self.layers =8
        self.dropout_prob = dropout
        self.img_size = (252, 252)
        self.hidden_size = 100
        self.conv1 = CDC(3,self.layers,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2 = CDC(self.layers,1,kernel_size=3,stride=1,padding=1)
        # Maxpool
        self.linear_1 = nn.Linear((self.img_size[0]//4 * self.img_size[1]//4), self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.linear_2 = nn.Linear(self.hidden_size, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inp):
        with torch.no_grad():
            cdc_out = self.cdc_net(inp)
        conv1 = self.conv1(cdc_out.repeat(1,3,1,1))
        relu1 = self.relu(conv1)
        maxpool = self.maxpool(relu1)
        conv2 = self.conv2(maxpool)
        relu2 = self.relu(conv2)
        maxpool2 = self.maxpool(relu2)
        linear_1 = self.linear_1(maxpool2.view(-1, self.img_size[0]//4 * self.img_size[1]//4))
        dropout = self.dropout(linear_1)
        linear_2 = self.linear_2(dropout)
        return self.sigmoid(linear_2),cdc_out