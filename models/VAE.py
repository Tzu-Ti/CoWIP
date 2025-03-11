import torch
from torch import Tensor
from models.Modules import *
from torch import nn

class Encoder(nn.Module):
    def __init__(self, in_size: Tuple = (192, 256), activation='leakyrelu') -> None:
        super(Encoder, self).__init__()
        inchannels = [64, 128, 256, 512, 1024]
        self.activation = activation
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 16, 7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            Activation(self.activation),
            nn.Conv2d(16, inchannels[0], 5, stride=2, padding=2),
            nn.BatchNorm2d(inchannels[0]),
            Activation(self.activation),
        )
        self.layers = self._make_layers(inchannels)
        self.bottle_neck = nn.Sequential(
            ResidualBlock(inchannels[-1], inchannels[-1], self.activation),
            ResidualBlock(inchannels[-1], inchannels[-1], self.activation),
        )
        self.feature_size = (in_size[0] // (2**(len(inchannels) + 1)), 
                             in_size[1] // (2**(len(inchannels) + 1)))
        
    def _make_layers(self, inchannels:List[int]):
        layers = nn.ModuleList()
        for i in range(len(inchannels) - 1):
            layers.append(nn.Sequential(
                nn.Conv2d(inchannels[i], inchannels[i+1], 3, stride=2, padding=1),
                nn.BatchNorm2d(inchannels[i+1]),
                Activation(self.activation),
                ResidualBlock(inchannels[i+1], inchannels[i+1], self.activation),
            ))
            print(f"Encoder layer info-> <layer:{i}, in_channels:{inchannels[i]}, out_channels:{inchannels[i+1]}>")
        return layers
    
    def forward(self, x):
        x = self.input_block(x)
        for layer in self.layers:
            x = layer(x)
        x = self.bottle_neck(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, activation='leakyrelu') -> None:
        super(Decoder, self).__init__()
        inchannels = [1024, 512, 256, 128, 64]
        self.activation = activation
        self.bottle_neck = nn.Sequential(
            ResidualBlock(inchannels[0], inchannels[0], self.activation),
            ResidualBlock(inchannels[0], inchannels[0], self.activation),
        )
        print("[MODEL] Create VAE Decoder:")
        self.layers = self._make_layers(inchannels)
        print(f"\tOut layer info-> <layer:0, in_channels:{inchannels[-1]}, out_channels:16>")
        print(f"\tOut layer info-> <layer:1, in_channels:16, out_channels:1>")
        print(f"\tOut layer info-> <layer:2, in_channels:1, out_channels:1>")
        print(f"\tOut layer info-> <output activation: None>")
        self.out_block = nn.Sequential(
            nn.ConvTranspose2d(inchannels[-1], 16, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(16),
            Activation(self.activation),
            nn.ConvTranspose2d(16, 4, 7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(4),
            Activation(self.activation),
            nn.Conv2d(4, 1, 3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            Activation(self.activation),
            nn.Conv2d(1, 1, 3, stride=1, padding=1),
        )
        print("[MODEL] VAE Decoder created.")


    def _make_layers(self, inchannels:List[int]):
        layers = nn.ModuleList()
        for i in range(len(inchannels) - 1):
            layers.append(nn.Sequential(
                ResidualBlock(inchannels[i], inchannels[i], self.activation),
                nn.ConvTranspose2d(inchannels[i], inchannels[i+1], 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(inchannels[i+1]),
                Activation(self.activation),
            ))
            print(f"\tDecoder layer info-> <layer:{i}, in_channels:{inchannels[i]}, out_channels:{inchannels[i+1]}>")
        return layers
    
    def forward(self, x):
        x = self.bottle_neck(x)
        for layer in self.layers:
            x = layer(x)
        return self.out_block(x)
    
class VAE(nn.Module):
    def __init__(self, activation='leakyrelu') -> None:
        super(VAE, self).__init__()
        self.encoder = Encoder(activation=activation)
        self.decoder = Decoder(activation=activation)
        self.feature_size = self.encoder.feature_size
        self.latent_dim = 1024*self.feature_size[0]*self.feature_size[1]
        self.fc_mu = nn.Sequential(
            nn.Linear(self.latent_dim , self.latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.latent_dim , self.latent_dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim , self.latent_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.latent_dim)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z = self.fc(z)
        z = z.view(-1, 1024, self.feature_size[0], self.feature_size[1])
        x = self.decoder(z)
        return x, mu, logvar, z

class VAE_Finetune(nn.Module):
    def __init__(self, activation='leakyrelu') -> None:
        super(VAE_Finetune, self).__init__()
        self.encoder = Encoder(activation=activation)
        self.feature_size = self.encoder.feature_size
        self.latent_dim = 1024*self.feature_size[0]*self.feature_size[1]
        self.fc_mu = nn.Sequential(
            nn.Linear(self.latent_dim , self.latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.latent_dim , self.latent_dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim , self.latent_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.latent_dim)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z = self.fc(z)
        z = z.view(-1, 1024, self.feature_size[0], self.feature_size[1])

        return z