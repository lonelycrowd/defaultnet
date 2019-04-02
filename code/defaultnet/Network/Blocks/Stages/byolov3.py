import torch
import torch.nn as nn

from ..conv import Conv2dBatchLeaky,Conv2dBatchDefLeaky


class Head(nn.Module):
    custom_layers = ()
    def __init__(self, nchannels, nanchors, nclasses):
        super().__init__()
        mid_nchannels = 2 * nchannels
        layer_list = [
                Conv2dBatchLeaky(nchannels, mid_nchannels, 3, 1),
                nn.Conv2d(mid_nchannels, nanchors*(5+nclasses), 1, 1, 0),
                ]
        self.feature =  nn.Sequential(*layer_list)

    def forward(self, data):
        x = self.feature(data)
        return x
    
class DefHead(nn.Module):
    custom_layers = ()
    def __init__(self, nchannels, nanchors, nclasses):
        super().__init__()
        mid_nchannels = 2 * nchannels
        layer_list = [
                Conv2dBatchDefLeaky(nchannels, mid_nchannels, 3, 1),
                nn.Conv2d(mid_nchannels, nanchors*(5+nclasses), 1, 1, 0),
                ]
        self.feature =  nn.Sequential(*layer_list)

    def forward(self, data):
        x = self.feature(data)
        return x

