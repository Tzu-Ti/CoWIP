import torch
from torch import nn
from models.VAE import Decoder
from models.Encoder import ERC_Transformer
from models.Modules import AggregationBlock, CrossAggregationBlock, Adapter_Block
from functools import reduce

class CSIEncoder(nn.Module):
    def __init__(self, CONFIG:dict):
        super(CSIEncoder, self).__init__()
        """
        Full Model for CSI Reconstruction
        args:
            CONFIG: dict, configuration for ERC_Transformer and FullModel
            vae_latent_shape : [C, H, W]
        """
        print("[MODEL] Create Full Model")
        self.encoder = ERC_Transformer(**CONFIG['ERC_Transformer'])
        self.lat_shape = CONFIG['FullModel']['vae_latent_shape'] # [C, H, W]
        latent_dim = reduce(lambda x, y: x*y, self.lat_shape)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.fc = nn.Linear(latent_dim, latent_dim)
        if CONFIG['FullModel']['aggregation'] == 'cross':
            self.aggr = CrossAggregationBlock(**CONFIG['AggregationBlock'])
        else:
            self.aggr = AggregationBlock(**CONFIG['AggregationBlock'])

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, amp, pha, srcmask=None):
        if self.encoder.return_channel_stream:
            amp, pha, amp_channel, pha_channel = self.encoder(amp, pha, srcmask)
        else:
            amp, pha = self.encoder(amp, pha, srcmask)
        out = self.aggr(amp, pha)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        z = self.reparameterize(mu, log_var)
        z = self.fc(z)
        z = z.reshape(-1, self.lat_shape[0], self.lat_shape[1], self.lat_shape[2])
        
        if self.encoder.return_channel_stream:
            return z, mu, log_var, amp_channel, pha_channel
        else:
            return z, mu, log_var

class FullModel_Finetune(nn.Module):
    def __init__(self, CONFIG:dict, decoder:Decoder):
        super(FullModel_Finetune, self).__init__()
        """
        Full Model for CSI Reconstruction
        args:
            CONFIG: dict, configuration for ERC_Transformer and FullModel
            decoder: nn.Module, VAE Decoder
            vae_latent_shape : [C, H, W]
        """
        print("[MODEL] Create Full Model")
        self.encoder = ERC_Transformer(**CONFIG['ERC_Transformer'])
        self.encoder.return_channel_stream = False
        self.encoder.encoder.return_channel_stream = False
        self.decoder = decoder
        self.lat_shape = CONFIG['FullModel']['vae_latent_shape'] # [C, H, W]
        latent_dim = reduce(lambda x, y: x*y, self.lat_shape)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.fc = nn.Linear(latent_dim, latent_dim)
        if CONFIG['FullModel']['aggregation'] == 'cross':
            self.aggr = CrossAggregationBlock(**CONFIG['AggregationBlock'])
        else:
            self.aggr = AggregationBlock(**CONFIG['AggregationBlock'])

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, amp, pha, srcmask=None):
        amp, pha = self.encoder(amp, pha, srcmask)
        out = self.aggr(amp, pha)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        z = self.reparameterize(mu, log_var)
        z = self.fc(z)
        z = z.reshape(-1, self.lat_shape[0], self.lat_shape[1], self.lat_shape[2])

        out = self.decoder(z)

        return out, mu, log_var, z