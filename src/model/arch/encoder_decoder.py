import torch
import torch.nn as nn
import torch.nn.functional as F
from model.arch.modules import ConvBlock, ResConvBlock

class EncoderBase(nn.Module):
    '''
        Base class for encoder
        Specify print_layer to see output shape at each layer
    '''
    def __init__(self, line_len=128, size_multiplier=128, print_layer=False, flatten=True, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential()

        self.print_layer = print_layer
        self.line_len = line_len
        self.size_multiplier = size_multiplier
        self.reshape_size = 16 if self.line_len==128 else 32
        self.linear_dim = self.size_multiplier * self.reshape_size
        self.flatten = flatten

    def reset_reshape_size(self):
        if len(self.encoder) < 2:
            return
        n_layer = max(len(self.encoder), 1)
        self.reshape_size = (self.line_len) // (pow(2, n_layer)) # get input length after encoding
        self.linear_dim = self.size_multiplier * self.reshape_size
        if self.flatten: # flatten encoder output
            self.encoder.append(nn.Flatten())

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.print_layer:
            print("---Encoder---")
            print(f"Input: {x.size()}")
            for i, layer in enumerate(self.encoder):
                x = layer(x)
                print(f"Layer {i+1}: {x.size()}")
            return x
        return self.encoder(x)


class DecoderBase(nn.Module):
    '''
        Base class for decoder
        Specify print_layer to see output shape at each layer
    '''
    def __init__(self, line_len=128, size_multiplier=128, print_layer=False, flattened=True, **kwargs):
        super().__init__()

        self.print_layer = print_layer
        self.line_len = line_len
        self.size_multiplier = size_multiplier
        self.reshape_size = 16 if self.line_len==128 else 32
        self.linear_dim = self.size_multiplier * self.reshape_size
        self.flattened = flattened
        
        # if self.flattened:
        #     self.decoder_linear = nn.Sequential()
        self.decoder_pre = nn.Sequential()
        self.decoder_conv = nn.Sequential()

    def reset_reshape_size(self, dims_in, final_layer_offset=0):
        if len(self.decoder_conv) < 1:
            return
        self.reshape_size = (self.line_len) // (pow(2, len(self.decoder_conv)-final_layer_offset)) # get input length after encoding
        self.linear_dim = self.size_multiplier * self.reshape_size
        if self.flattened: # if input is flattened, add linear layer to unflatten
            self.decoder_pre = nn.Sequential(
                nn.Linear(dims_in, self.linear_dim, bias=False),
                nn.ReLU()
            )
        elif dims_in != -1:
            self.decoder_pre.append(nn.Conv1d(dims_in, self.decoder_conv[0].conv.in_channels, 
                                              kernel_size=1, stride=1))

    def forward(self, x):
        if self.print_layer:
            print("---Decoder---")
            print(f"Input: {x.size()}")
            if len(self.decoder_pre) != 0:
                x = self.decoder_pre(x)
                print(f"Pre-conv: {x.size()}")
                if self.flattened: # if input is flattened, reshape to pass into conv layers
                    x = x.reshape((-1, self.size_multiplier, self.reshape_size))
                    print(f"Resized: {x.size()}")
            for i, layer in enumerate(self.decoder_conv):
                x = layer(x)
                print(f"Layer {i+1}: {x.size()}")
            return x.permute(0, 2, 1)
        
        if len(self.decoder_pre) != 0:
            x = self.decoder_pre(x)
            if self.flattened: # if input is flattened, add linear layer to unflatten
                x = x.reshape((-1, self.size_multiplier, self.reshape_size))
        out = self.decoder_conv(x).permute(0,2,1)
        return out
    

class Encoder(EncoderBase):
    def __init__(self, channels_in, block=ConvBlock, kernel_dims=[63,31,15], layer_dims=[16,32,64], **kwargs):
        super().__init__(size_multiplier=layer_dims[-1], **kwargs)

        blocks = [block] * len(kernel_dims)
        if isinstance(block, ResConvBlock):
            blocks[0] = ConvBlock
            
        layers = []
        ci = channels_in
        for k, c, b in zip(kernel_dims, layer_dims, blocks):
            layers.append(b(ci, c, k, **kwargs))
            ci = c
        self.encoder = nn.Sequential(*layers)
        self.reset_reshape_size()
    

class Decoder(DecoderBase):
    def __init__(self, dims_in, channels_out, block=ConvBlock, kernel_dims=[15,31,63], layer_dims=[64,32,16], **kwargs):
        super().__init__(size_multiplier=layer_dims[0], **kwargs)

        blocks = [block] * len(kernel_dims)
        if isinstance(block, ResConvBlock):
            blocks[-1] = ConvBlock
            
        layers = []
        for i, (k, c, b) in enumerate(zip(kernel_dims, layer_dims, blocks)):
            cout = channels_out if i==len(layer_dims)-1 else layer_dims[i+1]
            last_layer = True if i==len(layer_dims)-1 else False
            layers.append(b(c, cout, k, deconvolution=True, last_layer=last_layer, **kwargs))
        self.decoder_conv = nn.Sequential(*layers)
        
        self.reset_reshape_size(dims_in=dims_in)


class Encoder3Ls(EncoderBase):
    '''Encoder with 3 conv layers (63, 31, 15)'''
    def __init__(self, channels_in, **kwargs):
        super().__init__(size_multiplier=64, **kwargs)

        self.encoder = nn.Sequential(
            ConvBlock(channels_in, 16, 63, **kwargs),
            ConvBlock(16, 32, 31, **kwargs),
            ConvBlock(32, 64, 15, **kwargs),
        )
        self.reset_reshape_size()
    

class Decoder3Ls(DecoderBase):
    '''Decoder with 3 conv layers (15, 31, 63)'''
    def __init__(self, dims_in, channels_out, **kwargs):
        super().__init__(size_multiplier=64, **kwargs)

        self.decoder_conv = nn.Sequential(
            ConvBlock(64, 32, 15, deconvolution=True, **kwargs),
            ConvBlock(32, 16, 31, deconvolution=True, **kwargs),
            ConvBlock(16, channels_out, 63, deconvolution=True, last_layer=True, **kwargs)
        )
        self.reset_reshape_size(dims_in=dims_in)


class EncoderRes3L(EncoderBase):
    '''Encoder with 3 residual blocks (63, 31, 15)'''
    def __init__(self, channels_in, **kwargs):
        super().__init__(size_multiplier=128, **kwargs)

        self.encoder = nn.Sequential(
            ConvBlock(channels_in, 16, 31, **kwargs),
            ResConvBlock(16, 32, 15, **kwargs),
            ResConvBlock(32, 64, 15, **kwargs),
            ResConvBlock(64, 128, 15, **kwargs),
        )
        self.reset_reshape_size()
    

class DecoderRes3L(DecoderBase):
    '''Decoder with 3 residual blocks (15, 31, 63)'''
    def __init__(self, dims_in, channels_out, **kwargs):
        super().__init__(size_multiplier=128, **kwargs)

        self.decoder_conv = nn.Sequential(
            ResConvBlock(128, 64, 15, deconvolution=True, **kwargs),
            ResConvBlock(64, 32, 15, deconvolution=True, **kwargs),
            ResConvBlock(32, 16, 15, deconvolution=True, **kwargs),
            ConvBlock(16, channels_out, 31, deconvolution=True, last_layer=True,**kwargs),
        )
        self.reset_reshape_size(dims_in=dims_in)


class EncoderRes3Ls(EncoderBase):
    def __init__(self, channels_in, **kwargs):
        super().__init__(size_multiplier=128, **kwargs)

        self.encoder = nn.Sequential(
            ConvBlock(channels_in, 16, 5, **kwargs),
            ResConvBlock(16, 32, 5, **kwargs),
            ResConvBlock(32, 64, 5, **kwargs),
            ResConvBlock(64, 128, 5, **kwargs),
        )
        self.reset_reshape_size()
    

class DecoderRes3Ls(DecoderBase):
    def __init__(self, dims_in, channels_out, **kwargs):
        super().__init__(size_multiplier=128, **kwargs)

        self.decoder_conv = nn.Sequential(
            ResConvBlock(128, 64, 5, deconvolution=True, **kwargs),
            ResConvBlock(64, 32, 5, deconvolution=True, **kwargs),
            ResConvBlock(32, 16, 5, deconvolution=True, **kwargs),
            ConvBlock(16, channels_out, 5, deconvolution=True, last_layer=True,**kwargs),
        )
        self.reset_reshape_size(dims_in=dims_in)