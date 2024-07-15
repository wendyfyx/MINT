import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm_layer(n_channels, n_norm_groups=1, norm_layer_type=None, eps=1e-5, **kwargs):
    if isinstance(norm_layer_type, str) and norm_layer_type.lower() == 'bn': # BatchNorm
        return nn.BatchNorm1d(n_channels, eps=eps)
    elif isinstance(norm_layer_type, str) and norm_layer_type.lower() == 'gn': # GroupNorm
        return nn.GroupNorm(n_norm_groups, n_channels, eps=eps)
    elif isinstance(norm_layer_type, str) and norm_layer_type.lower() == 'ln': # LayerNorm
        return nn.GroupNorm(1, n_channels, eps=eps)
    elif isinstance(norm_layer_type, str) and norm_layer_type.lower() == 'in': # InstanceNorm
        return nn.GroupNorm(n_channels, n_channels, eps=eps)
    else:
        return None # No normalization layer


class ConvBlock(nn.Module):
    '''Block of conv + non-linear'''
    def __init__(self, channels_in, channels_out, kernel_size, 
                 deconvolution=False, last_layer=False, 
                 norm_first=False, **kwargs):
        super().__init__()
        
        layers = []
        self.stride=2
        self.padding = (kernel_size-1)//2
        if deconvolution:
            conv = nn.ConvTranspose1d(channels_in, channels_out, kernel_size=kernel_size, 
                                           stride=self.stride, padding=self.padding, output_padding=1, bias=False)
        else:
            conv = nn.Conv1d(channels_in, channels_out, kernel_size=kernel_size, 
                                  stride=self.stride, padding=self.padding, bias=False)
        
        actv = nn.Tanh() if last_layer else nn.LeakyReLU(1e-2)
        norm = get_norm_layer(channels_out, **kwargs)
        
        layers.append(conv)
        if norm is not None:
            if norm_first:
                layers.extend([norm, actv])
            else:
                layers.extend([actv, norm])
        else:
            layers.append(actv)
        self.layers = nn.Sequential(*layers)
            
        
    def forward(self, x):
        out = self.layers(x)
        return out


class ResConvBlock(ConvBlock):
    '''
        Residual Blocks with Dilated Convolution
        MAIN: Conv -> BN -> ReLU -> Conv -> BN
        SKIP: Conv -> BN
        Ref: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
    '''
    def __init__(self, channels_in, channels_out, kernel_size, 
                 deconvolution=False, **kwargs):
        super().__init__(channels_in, channels_out, kernel_size,
                         deconvolution=deconvolution, **kwargs)

        # Second convolution layer
        if deconvolution:
            conv2 = nn.ConvTranspose1d(channels_out, channels_out, kernel_size=5, stride=1, padding=2, bias=False)
        else:
            conv2 = nn.Conv1d(channels_out, channels_out, kernel_size=5, stride=1, padding=2, bias=False)
        self.layers.append(conv2)
        norm2 = get_norm_layer(channels_out, **kwargs) # Second norm
        if norm2 is not None:
            self.layers.append(norm2)

        # Convolution Block for when input and output dimension are mismatched 
        # Use the same init param as the first conv layer
        # Ref: https://medium.com/analytics-vidhya/understanding-and-implementation-of-residual-networks-resnets-b80f9a507b9c
        skip_layers = []
        if deconvolution:
            skip_layers.append(nn.ConvTranspose1d(channels_in, channels_out, kernel_size=kernel_size, 
                                                       stride=self.stride, padding=self.padding, output_padding=1, bias=False))
        else:
            skip_layers.append(nn.Conv1d(channels_in, channels_out, kernel_size=kernel_size, 
                                              stride=self.stride, padding=self.padding, bias=False))
        norm3 = get_norm_layer(channels_out, **kwargs) # Third norm
        if norm3 is not None:
            skip_layers.append(norm3)
        self.sample_layers = torch.nn.Sequential(*skip_layers)
        self.final_relu = nn.ReLU(inplace=True) # Final activation after skip connection

    def forward(self, x):
        residual = x

        # Main branch
        out = self.layers(x)

        # Skip branch
        residual = self.sample_layers(x)
        out += residual
        out = self.final_relu(out)
        return out
    

class SepConv1d(nn.Module):
    """A simple separable convolution implementation.
    
    The separable convlution is a method to reduce number of the parameters 
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, 
                                   groups=ni, bias=False)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SepTransposeConv1d(nn.Module):
    """A simple separable convolution implementation.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.ConvTranspose1d(ni, ni, kernel, stride, padding=pad, 
                                            output_padding=1, groups=ni, bias=False)
        self.pointwise = nn.ConvTranspose1d(ni, no, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

    
class SepConvBlock(nn.Module):
    '''Block of conv + non-linear'''
    def __init__(self, channels_in, channels_out, kernel_size, 
                 deconvolution=False, last_layer=False, 
                 norm_first=False, **kwargs):
        super().__init__()
        
        layers = []
        self.stride=2
        self.padding = (kernel_size-1)//2
        if deconvolution:
            conv = SepTransposeConv1d(channels_in, channels_out, 
                                      kernel_size, self.stride, self.padding)
        else:
            conv = SepConv1d(channels_in, channels_out, 
                             kernel_size, self.stride, self.padding)
        
        actv = nn.Tanh() if last_layer else nn.LeakyReLU(1e-2)
        norm = get_norm_layer(channels_out, **kwargs)
        
        layers.append(conv)
        if norm is not None:
            if norm_first:
                layers.extend([norm, actv])
            else:
                layers.extend([actv, norm])
        else:
            layers.append(actv)
        self.layers = nn.Sequential(*layers)
            
        
    def forward(self, x):
        out = self.layers(x)
        return out