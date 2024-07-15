import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    '''1D Convolutional VAE class with custom encoder & decoder'''
    def __init__(self, hidden_dim, encoder, decoder, mu=None, logvar=None, log_scale=None):
        super().__init__()

        self.encoder = encoder
        self.decoder =decoder
        

        self.mu = nn.Linear(self.encoder.linear_dim, hidden_dim, bias=False) if mu is None else mu
        self.logvar = nn.Linear(self.encoder.linear_dim, hidden_dim, bias=False) if logvar is None else logvar

        self.log_scale = nn.Parameter(torch.Tensor([0.0])) if log_scale is None else log_scale

    def encode(self, x):
        '''Encoder q(z|x) given input x, generate parameters for q(z|x)'''
        x = x.double()
        out = self.encoder(x)
        return self.mu(out), self.logvar(out)
    
    def encode_z(self, x):
        # encode x, get q(z|x) parameters
        mu, logvar = self.encode(x)
        
        # reparametrize, sample z from q(z|x)
        z, std = self.reparameterize(mu, logvar)

        return z, mu, std

    def reparameterize(self, mu, logvar):
        '''
            Reparameterize trick to allow backpropagation through mu, std.
            We can't backpropagate that if we sample mu and std directly.
            z is sampled from q
        '''
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        z = mu + (eps * std) # sampling as if coming from the input space
        return z, std

    def decode(self, z):
        '''Decode given z, get p(x|z) parameters'''
        return self.decoder(z)
            
    def forward(self, x, y=None):
        '''To get output and z, use this for evaluation'''
        
        # encode x, get q(z|x) parameters
        mu, logvar = self.encode(x)
        
        # reparametrize, sample z from q(z|x)
        z, std = self.reparameterize(mu, logvar)

        # decode, get p(x|z) parameters
        x_hat = self.decode(z).view(x.size())
        return x_hat, z, mu, std