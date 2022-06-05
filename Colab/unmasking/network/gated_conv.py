import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/zhaoyuzhi/deepfillv2/blob/master/deepfillv2/network_module.py#L82
class GatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, padding_type: str = "reflect", activation: str = "elu", norm: str = "none", sn: bool = False):
        super().__init__()

        # Initialize the padding scheme
        if padding_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif padding_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif padding_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(padding_type)

        # Initialize the normalization type
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_channels)
        #  elif norm == "ln":
        #      self.norm = LayerNorm(out_channels)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == "relu":
            self.activation = nn.ReLU(inplace = True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace = True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.sigmoid = torch.nn.Sigmoid()

        # Initialize the convolution layers
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.x_conv_layers = nn.ModuleList([
            self.conv2d
        ])
        if self.activation is not None:
            self.x_conv_layers.append(self.activation)
        self.x_branch = nn.Sequential(self.x_conv_layers)

        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        if sn:
            self.conv2d = nn.utils.spectral_norm(self.conv2d)
            self.mask_conv2d = nn.utils.spectral_norm(self.mask_conv2d)

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)

        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)

        x = conv * gated_mask
        return x


class GatedDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, padding_type: str = "zero", activation: str = "lrelu", norm: str = "none", sn: bool = True, scale_factor: float = 2.0):
        super().__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, padding_type, activation, norm, sn)

    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = "nearest")
        x = self.gated_conv2d(x)
        return x
