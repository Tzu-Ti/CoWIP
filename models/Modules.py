from typing import Union, Tuple, List, Callable, Optional
from models.MHMA import MultiHeadMixAttention
from torch.nn import functional as F
from utils import Activation
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch
import math
import copy

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 activation:str = 'SiLU', dropout: float = 0.1,
                 output_activation: bool = True):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_activation = output_activation
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1),
            nn.Dropout(dropout),
            nn.BatchNorm2d(out_channels),
            Activation(activation),
            nn.Conv2d(out_channels, out_channels, 3, stride=1,
                      padding=1, groups=out_channels//4),
            nn.Dropout(dropout),
            nn.BatchNorm2d(out_channels),
            Activation(activation),
            nn.Conv2d(out_channels, out_channels, 1, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        if self.in_channels != self.out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels,
                          3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        self.activation = Activation(activation)

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.in_channels != self.out_channels:
            residual = self.downsample(residual)
        out += residual
        
        if self.output_activation:
            return self.activation(out)
        else:
            return out

class Self_Attn_Block(nn.Module):
    def __init__(self, 
                 d_model: int, nhead: int, dropout: float = 0.1,
                 batch_first: bool = True, norm_first: bool = True, layer_norm_eps: float = 1e-5,
                 bias: bool = True, device=None, dtype=None):
        super(Self_Attn_Block, self).__init__()
        self.norm_first = norm_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.MA = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, **factory_kwargs)

    def _SA_block(self, x: Tensor, src_mask: Optional[Tensor], src_key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.MA(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False, is_causal=is_causal)[0]
        return self.dropout(x)
    
    
    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:

        if self.norm_first:
            x = x + self._SA_block(self.l_norm(x), src_mask, src_key_padding_mask, is_causal)
        else:
            x = self.l_norm(x + self._SA_block(x, src_mask, src_key_padding_mask, is_causal))
        return x
    
class Cross_Attn_Block(nn.Module):
    def __init__(self, 
                 d_model: int, nhead: int, dropout: float = 0.1,
                 batch_first: bool = True, norm_first: bool = True, layer_norm_eps: float = 1e-5,
                 bias: bool = True, device=None, dtype=None):
        super(Cross_Attn_Block, self).__init__()
        self.norm_first = norm_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.MA = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, **factory_kwargs)

    def _CA_block(self, q: Tensor, k: Tensor, v: Tensor,
                  src_mask: Optional[Tensor], src_key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.MA(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False, is_causal=is_causal)[0]
        return self.dropout(x)
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        if self.norm_first:
            q = q + self._CA_block(self.l_norm(q), self.l_norm(k), self.l_norm(v), src_mask, src_key_padding_mask, is_causal)
        else:
            q = self.l_norm(q + self._CA_block(q, k, v, src_mask, src_key_padding_mask, is_causal))
        return self.dropout(q)
    
class Mix_Attn_Block(nn.Module):
    def __init__(self, 
                 d_model: int, nhead: int, dropout: float = 0.1,
                 batch_first: bool = True, norm_first: bool = True, layer_norm_eps: float = 1e-5,
                 bias: bool = True, device=None, dtype=None):
        super(Mix_Attn_Block, self).__init__()
        self.norm_first = norm_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.a_l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.b_l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.MA = MultiHeadMixAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)

    def _MA_block(self, a: Tensor, b: Tensor,
                  src_mask: Optional[Tensor], 
                  a_key_padding_mask: Optional[Tensor], b_key_padding_mask: Optional[Tensor]) -> Tensor:
        a, b = self.MA(a, b, 
                    attn_mask = src_mask, 
                    a_key_padding_mask=a_key_padding_mask, 
                    b_key_padding_mask=b_key_padding_mask)
        return self.dropout(a), self.dropout(b)
    
    def forward(self, a: Tensor, b: Tensor, 
                src_mask: Optional[Tensor] = None,
                a_key_padding_mask: Optional[Tensor] = None, b_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if self.norm_first:
            new_a, new_b = self._MA_block(self.a_l_norm(a), self.b_l_norm(b), src_mask, a_key_padding_mask, b_key_padding_mask)
            a, b = a + new_a, b + new_b
        else:
            new_a, new_b = self._MA_block(a, b, src_mask, a_key_padding_mask, b_key_padding_mask)
            a, b = self.a_l_norm(a + new_a), self.b_l_norm(b + new_b)
        return self.dropout(a), self.dropout(b)

class Rev_Attn_Block(nn.Module):
    def __init__(self, 
                 d_model: int, dim_feedforward: int,
                 dropout: float = 0.1, activation: str = 'leakyrelu',
                 use_reverse: bool = True):
        super(Rev_Attn_Block, self).__init__()
        self.amp_sigmoid = nn.Sigmoid()
        self.pha_sigmoid = nn.Sigmoid()
        self.amp_ff = FF_Block(d_model, dim_feedforward, dropout, activation)
        self.pha_ff = FF_Block(d_model, dim_feedforward, dropout, activation)
        self.amp_contrasive_linear = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            Activation(activation),
            nn.Linear(d_model//2, d_model//4),
        )
        self.pha_contrasive_linear = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            Activation(activation),
            nn.Linear(d_model//2, d_model//4),
        )
        self.use_reverse = use_reverse
        self.infer_mode = False

    def forward(self, amp:torch.Tensor, pha:torch.Tensor, src_amp:torch.Tensor, src_pha:torch.Tensor):
        if self.use_reverse:
            rev_amp = src_amp * self.amp_sigmoid(-amp)
            rev_pha = src_pha * self.pha_sigmoid(-pha)
            amp = self.amp_ff(rev_amp)
            pha = self.pha_ff(rev_pha)
            if not self.infer_mode:
                rev_amp = self.amp_contrasive_linear(rev_amp)
                rev_pha = self.pha_contrasive_linear(rev_pha)
        else:
            rev_amp = None
            rev_pha = None
            amp = self.amp_ff(amp)
            pha = self.pha_ff(pha)
            
        return amp, pha, rev_amp, rev_pha

class FF_Block(nn.Module):
    def __init__(self, 
                 d_model: int, dim_feedforward: int=2048,
                 dropout: float = 0.1, activation: str = 'relu',
                 norm_first: bool = True, layer_norm_eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super(FF_Block, self).__init__()
        self.norm_first = norm_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.activation = Activation(activation)

    def _FF_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.norm_first:
            x = x + self._FF_block(self.l_norm(x))
        else:
            x = self.l_norm(x + self._FF_block(x))
        return x

class MultiScale_Convolution_Block(nn.Module):
    def __init__(self, 
                 d_model: int, kernel_sizes: List[int], dim_feedforward: int=2048, 
                 dropout: float = 0.1, activation: str = "relu",norm_first: bool = True,
                 layer_norm_eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super(MultiScale_Convolution_Block, self).__init__()
        self.norm_first = norm_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = dropout
        self.conv_blocks = self._make_layers(d_model, kernel_sizes, activation)
        self.weight_FF = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs),
            Activation(activation),
            nn.Linear(dim_feedforward, 1, bias=bias, **factory_kwargs)
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def _make_layers(self, d_model: int, kernel_sizes: List[int], activation:str) -> nn.ModuleList:
        layers = []
        for kernel_size in kernel_sizes:
            layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(d_model),
                nn.Dropout(self.dropout),
                Activation(activation)
                ))
        return nn.ModuleList(layers)
    
    def _msc_forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: torch.Tensor, shape [batch_size, d_model, seq_len]
        Returns:
            out: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        multiscale_features = None
        for conv in self.conv_blocks:
            if multiscale_features is None:
                multiscale_features = conv(x).unsqueeze(-1)
            else:
                multiscale_features = torch.cat((multiscale_features, conv(x).unsqueeze(-1)), dim=-1)
        # multiscale_features.shape: [batch_size, d_model, seq_len, len(kernel_sizes)]
        multiscale_features = multiscale_features.permute(0, 2, 1, 3)
        # multiscale_features.shape: [batch_size, seq_len, d_model, len(kernel_sizes)]
        weights = self.softmax(self.weight_FF(multiscale_features.transpose(-1, -2)).squeeze(-1))
        # weights.shape: [batch_size, seq_len, len(kernel_sizes)]
        out = torch.matmul(multiscale_features, weights.unsqueeze(-1)).squeeze(-1)
        # matmul = [n, 51, 1536, 5] * [n, 51, 5, 1]  -> [n, 51, 1536]
        return out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: torch.Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            out: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        if self.norm_first:
            out = inputs + self._msc_forward(self.l_norm(inputs).transpose(1, 2))
        else:
            out = self.l_norm(inputs + self._msc_forward(inputs.transpose(1, 2)))
        return out

class Adapter_Block(nn.Module):
    def __init__(self, in_length: int = 51, in_channels: int = 3072, 
                 out_length: int = 12, out_channels: int = 1024, 
                 activation: str = 'GELU', dropout: float = 0.1):
        super(Adapter_Block, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            Activation(activation),
            nn.Dropout(dropout),
        )
        self.CA_block = Cross_Attn_Block(out_channels, 8, dropout=dropout)
        self.ff_block = FF_Block(out_channels, 2048, dropout=dropout, activation=activation)
    
    def forward(self, Q: torch.Tensor, amp: torch.Tensor, pha: torch.Tensor):
        KV = torch.cat((amp, pha), dim=2)
        KV = self.in_proj(KV)
        Q = self.CA_block(Q, KV, KV)
        Q = self.ff_block(Q)
        return Q

class CrossAggregationBlock(nn.Module):
    def __init__(self, 
                 in_length: int = 51, in_channels: int = 3072,
                 out_length: int = 12, out_channels: int = 1024,
                 activation: str = 'GELU', dropout: float = 0.1):
        super(CrossAggregationBlock, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_channels*2, out_channels),
            nn.LayerNorm(out_channels),
            Activation(activation),
            nn.Dropout(dropout),
        )
        self.SA_block = Self_Attn_Block(out_channels, 8, dropout=dropout)
        self.ff_in_block = FF_Block(out_channels, 2048, dropout=dropout, activation=activation)
        self.CA_block = Cross_Attn_Block(out_channels, 8, dropout=dropout)
        self.ff_block = FF_Block(out_channels, 2048, dropout=dropout, activation=activation)
        self.out_container = nn.Parameter(torch.empty(out_length, out_channels), requires_grad=True)
        nn.init.xavier_uniform_(self.out_container, gain=1.0)
    
    def forward(self, amp: torch.Tensor, pha: torch.Tensor):
        csi = torch.cat((amp, pha), dim=2)
        csi = self.in_proj(csi)
        csi = self.SA_block(csi)
        csi = self.ff_in_block(csi)
        Q = self.out_container.unsqueeze(0).repeat(csi.size(0), 1, 1)
        Q = self.CA_block(Q, csi, csi)
        Q = self.ff_block(Q)
        Q = Q.transpose(1, 2).reshape(Q.size(0), -1)
        return Q

class AggregationBlock(nn.Module):
    def __init__(self, 
                 in_length: int = 51, in_channels: int = 3072,
                 out_length: int = 12, out_channels: int = 1024,
                 activation: str = 'SiLU', dropout: float = 0.1):
        super(AggregationBlock, self).__init__()
        self.amp_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, stride=2, padding=0),
            nn.BatchNorm1d(out_channels),
            Activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            Activation(activation),
            nn.Dropout(dropout),
        )
        self.pha_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, stride=2, padding=0),
            nn.BatchNorm1d(out_channels),
            Activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            Activation(activation),
            nn.Dropout(dropout),
        )
        self.aggregation = nn.Sequential(
            nn.Conv1d(out_channels*2, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            Activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            Activation(activation),
            nn.Dropout(dropout),
        )

    def forward(self, amp: torch.Tensor, pha: torch.Tensor):
        amp, pha = amp.transpose(1, 2), pha.transpose(1, 2)
        amp = self.amp_block(amp)
        pha = self.pha_block(pha)
        out = self.aggregation(torch.cat((amp, pha), dim=1))
        out = out.view(out.size(0), -1)

        return out
    
class ProjectionBlock(nn.Module):
    def __init__(self,
                 in_length: int,
                 RxTx_num: int,
                 in_channels: int, out_channels: int,
                 activation: str = 'relu', dropout: float = 0.1):
        super(ProjectionBlock, self).__init__()
        self.in_length = in_length
        self.RxTx_num = RxTx_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_projection = nn.Sequential(
            nn.AvgPool1d(4, 2, 1),
            nn.Linear(in_channels//2, out_channels),
            nn.LayerNorm(out_channels),
            Activation(activation),
            )
        
        if self.in_length == 153:
            self.pool_3 = nn.AvgPool1d(3, 3, 0)
            self.pool_5 = nn.AvgPool1d(5, 3, 1)
            self.temporal_projection = nn.Sequential(
                nn.Linear(in_length//3, in_length//3),
                nn.LayerNorm(in_length//3),
                Activation(activation),
                )
            
        elif self.in_length == 51:
            self.pool_3 = nn.AvgPool1d(3, 1, 1)
            self.pool_5 = nn.AvgPool1d(5, 1, 2)
            self.temporal_projection = nn.Sequential(
                nn.Linear(in_length, in_length),
                nn.LayerNorm(in_length),
                Activation(activation),
                )
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.in_length*self.RxTx_num, self.in_channels)
        x = self.channel_projection(x)
        x = x.view(-1, self.in_length, self.RxTx_num, self.out_channels)

        if self.in_length == 51:
            x = x.permute(0, 3, 2, 1).contiguous()
            x = x.view(-1, self.out_channels*self.RxTx_num, self.in_length)
            x = (x*4 + self.pool_3(x)*2 + self.pool_5(x)*1)/7
            x = self.temporal_projection(x)
            x = x.view(-1, self.out_channels, self.RxTx_num, self.in_length)
            x = x.permute(0, 3, 2, 1).contiguous()
            x = x.view(-1, self.in_length, self.RxTx_num*self.out_channels)

        elif self.in_length == 153:
            x = x.permute(0, 3, 2, 1).contiguous()
            x = x.view(-1, self.out_channels*self.RxTx_num, self.in_length)
            x = (x[:, :, 1::3]*4 + self.pool_3(x)*2 + self.pool_5(x)*1)/7
            x = self.temporal_projection(x)
            x = x.view(-1, self.out_channels, self.RxTx_num, self.in_length//3)
            x = x.permute(0, 3, 2, 1).contiguous()
            x = x.view(-1, self.in_length//3, self.RxTx_num*self.out_channels)

        return self.dropout(x)
        
class Spacial_Temporal_Embedding(nn.Module):
    def __init__(self, d_model: int, shape_input: List[int], npy_num: int = 3):
        """
        Args:
            d_model: int, embedding dimension
            d_input: List[int], input shape: [length, RxTx_num, subcarrier_num]
            k: int, number of gaussian distributions
        """
        super(Spacial_Temporal_Embedding, self).__init__()
        print("[MODEL] Create Spacial_Temporal_Embedding")
        print(f"\tEmbedding info-> <d_model:{d_model}, input_shape:{shape_input}>")
        print(f"\tEmbedding info-> <output shape:[{shape_input[0]}, {d_model}]>")
        self.d_model = d_model
        self.length = shape_input[0]
        self.RxTx_num = shape_input[1]
        self.d_input = shape_input[2]

        self.amp_projection = ProjectionBlock(self.length*npy_num, self.RxTx_num, self.d_input, d_model//self.RxTx_num)
        self.pha_projection = ProjectionBlock(self.length*npy_num, self.RxTx_num, self.d_input, d_model//self.RxTx_num)

        self.amp_embedding = nn.Parameter(torch.randn(self.length*self.RxTx_num, d_model//self.RxTx_num), requires_grad=True)
        self.pha_embedding = nn.Parameter(torch.randn(self.length*self.RxTx_num, d_model//self.RxTx_num), requires_grad=True)

    def forward(self, amp:torch.Tensor, pha:torch.Tensor):
        amp = self.amp_projection(amp).view(-1, self.length*self.RxTx_num, self.d_model//self.RxTx_num)
        pha = self.pha_projection(pha).view(-1, self.length*self.RxTx_num, self.d_model//self.RxTx_num)
        amp = amp + self.amp_embedding
        pha = pha + self.pha_embedding
        amp = amp.view(-1, self.length, self.d_model)
        pha = pha.view(-1, self.length, self.d_model)
        return amp, pha
    
class Gaussian_Range_Embedding(nn.Module):
    def __init__(self, d_model: int, shape_input: List[int], k: int = 10, npy_num: int = 3):
        """
        Args:
            d_model: int, embedding dimension
            shape_input: List[int], input shape: [length, RxTx_num, subcarrier_num]
            k: int, number of gaussian distributions
        """
        super(Gaussian_Range_Embedding, self).__init__()
        print("[MODEL] Create Gaussian_Range_Embedding")
        print(f"\tEmbedding info-> <d_model:{d_model}, input_shape:{shape_input}, k:{k}>")
        print(f"\tEmbedding info-> <output shape:[{shape_input[0]}, {d_model}]>")
        self.d_model = d_model
        self.length = shape_input[0]
        self.RxTx_num = shape_input[1]
        self.d_input = shape_input[2]
        self.k = k

        self.amp_projection = ProjectionBlock(self.length*npy_num, self.RxTx_num, self.d_input, d_model//self.RxTx_num)
        self.pha_projection = ProjectionBlock(self.length*npy_num, self.RxTx_num, self.d_input, d_model//self.RxTx_num)

        self.amp_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.k, d_model), requires_grad=True),
            gain=1.0
        )
        self.pha_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.k, d_model), requires_grad=True),
            gain=1.0
        )
        self.amp_mu = nn.Parameter(torch.linspace(0, self.length, self.k, requires_grad=True))
        self.amp_std = nn.Parameter(torch.ones(self.k, requires_grad=True) * self.length / 3)
        self.pha_mu = nn.Parameter(torch.linspace(0, self.length, self.k, requires_grad=True))
        self.pha_std = nn.Parameter(torch.ones(self.k, requires_grad=True) * self.length / 3)
        self.positions = torch.arange(0, self.length, 1).float().unsqueeze(1).repeat(1, self.k)
        self.positions = nn.Parameter(self.positions, requires_grad=False)
        self.const = -0.5 * torch.log(torch.tensor(2 * torch.pi))
        self.const = nn.Parameter(self.const, requires_grad=False)
    
    def _ln_pdf(self, pos: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        a = (pos - mu) / std
        ln_p = -torch.log(std) +self.const - 0.5 * a**2
        return nn.functional.softmax(ln_p, dim=1)
    
    def forward(self, amp:torch.Tensor, pha:torch.Tensor):
        amp = self.amp_projection(amp)
        pha = self.pha_projection(pha)

        amp_pdf = self._ln_pdf(self.positions, self.amp_mu, self.amp_std)
        pha_pdf = self._ln_pdf(self.positions, self.pha_mu, self.pha_std)
        amp_embedding = torch.matmul(amp_pdf, self.amp_embedding)
        pha_embedding = torch.matmul(pha_pdf, self.pha_embedding)

        return amp + amp_embedding, pha + pha_embedding

        
