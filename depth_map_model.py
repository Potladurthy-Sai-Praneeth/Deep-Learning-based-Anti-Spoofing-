import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class FineTuneDepthAnything(nn.Module):
    '''
    A class to fine-tune the Depth-Anything model for depth estimation. 
    The model is loaded from the Hugging Face model hub and only the last few layers are trained.
    Args :
        device : torch.device
            The device on which the model is loaded.
    Returns :   
        depth_anything : torch.tensor
            The depth estimation of the given input image.
    
    The input is a 3-channel RGB image and the output is a 1-channel depth map.
    '''
    def __init__(self, device):
        super(FineTuneDepthAnything, self).__init__()
        self.depth_anything = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        for name,param in self.depth_anything.named_parameters():
            if 'head' in name or 'neck.fusion_stage.layers.2.residual_layer' in name or 'neck.fusion_stage.layers.3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.depth_anything = self.depth_anything.to(device)
                
    def forward(self, inp):
        # print(f'inp shape: {inp.shape}')
        return self.depth_anything(inp).predicted_depth.unsqueeze(1)