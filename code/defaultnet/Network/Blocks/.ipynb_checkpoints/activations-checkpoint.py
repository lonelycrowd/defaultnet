import torch
import torch.nn as nn
import torch.nn.functional as F

class Scale(nn.Module):
    def __init__(self, nchannels, bias=True, init_scale=1.0):
        super().__init__()
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.nchannels = nchannels
        self.weight = nn.Parameter(torch.Tensor(1, nchannels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, nchannels, 1, 1))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale=1.0):
        self.weight.data.fill_(init_scale)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        # See the autograd section for explanation of what happens here.
        y = x * self.weight
        if self.bias is not None:
            y += self.bias
        return y

    def __repr__(self):
        s = '{} ({}, {})'
        return s.format(self.__class__.__name__, self.nchannels, self.bias is not None)


class ScaleReLU(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.scale = Scale(nchannels) 
        self.relu = nn.ReLU(inplace=True)
        self.nchannels = nchannels

    def forward(self, x):
        x1 = self.scale(x)
        y = self.relu(x1)
        return y

    def __repr__(self):
        s = '{name} ({nchannels})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class PPReLU(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.scale1 = Scale(nchannels, bias=False, init_scale=1.0) 
        self.scale2 = Scale(nchannels, bias=False, init_scale=0.1) 
        self.nchannels = nchannels

    def forward(self, x):
        x1 = self.scale1(x)
        x2 = self.scale2(x)
        y = torch.max(x1, x2)
        return y

    def __repr__(self):
        s = '{name} ({nchannels})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class PLU(nn.Module):
    """
    y = max(alpha*(x+c)−c, min(alpha*(x−c)+c, x))
    from PLU: The Piecewise Linear Unit Activation Function
    """
    def __init__(self, alpha=0.1, c=1):
        super().__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, x):
        x1 = self.alpha*(x + self.c) - self.c
        x2 = self.alpha*(x - self.c) + self.c
        min1 = torch.min(x2, x)
        min2 = torch.max(x1, min1)
        return min2

    def __repr__(self):
        s = '{name} ({alhpa}, {c})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class CReLU(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.scale = Scale(2*nchannels) 
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = nchannels
        self.out_channels = 2*nchannels

    def forward(self, x):
        x1 = torch.cat((x, -x), 1)
        x2 = self.scale(x1)
        y = self.relu(x2)
        return y

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class L2Norm(nn.Module):
    def __init__(self, nchannels, bias=True):
        super().__init__()
        self.scale = Scale(nchannels, bias=bias) 
        self.nchannels = nchannels
        self.eps = 1e-6

    def forward(self, x):
        #norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x = torch.div(x,norm)
        l2_norm = x.norm(2, dim=1, keepdim=True) + self.eps
        x_norm = x.div(l2_norm)
        y = self.scale(x_norm)
        return y

    def __repr__(self):
        s = '{name} ({nchannels})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
