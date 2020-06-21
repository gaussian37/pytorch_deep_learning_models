import torch.nn as nn
import torch

# DepthWise Separable Convolution
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        # depthwise convolution
        # depthwise convolution MUST have groups size same with the number of input channels.
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups = in_channels,
            bias = bias
        )
        
        # pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1,
            bias=bias
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x