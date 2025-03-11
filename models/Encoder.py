from typing import Union, Tuple, List, Callable, Optional
from torch.nn import functional as F
from models.Modules import *
from utils import Activation
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch
import math
import copy

class TemporalStream(nn.Module):
    def __init__(self,
                 d_model: int, n_heads: int, kernel_sizes: List[int], dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = 'gelu',
                 use_mix_attn: bool = False) -> None:
        super(TemporalStream, self).__init__()
        print("[MODEL] Create TemporalStream")
        print(f"\tTemporalStream info -> attn_type: {'Mix' if use_mix_attn else 'Cross'} Attention")
        print(f"\tTemporalStream info -> <d_model:{d_model}, n_heads:{n_heads}, kernel_sizes:{kernel_sizes}>")
        print(f"\tTemporalStream info -> <dim_feedforward:{dim_feedforward}, dropout:{dropout}, activation:{activation}>")
        self.use_mix_attn = use_mix_attn
        if not use_mix_attn:
            self.amp_ca = Cross_Attn_Block(d_model, n_heads, dropout)
            self.pha_ca = Cross_Attn_Block(d_model, n_heads, dropout)
        else:
            self.mix_attn = Mix_Attn_Block(d_model, n_heads, dropout)

        self.amp_msc = MultiScale_Convolution_Block(d_model, kernel_sizes, dim_feedforward, dropout, activation)
        self.pha_msc = MultiScale_Convolution_Block(d_model, kernel_sizes, dim_feedforward, dropout, activation)

    def _temporal_block(self, amp, pha):
        if not self.use_mix_attn:
            amp = self.amp_msc(self.amp_ca(amp, pha, pha))
            pha = self.pha_msc(self.pha_ca(pha, amp, amp))
        else:
            amp, pha = self.mix_attn(amp, pha)
            amp = self.amp_msc(amp)
            pha = self.pha_msc(pha)

        return amp, pha
    
    def forward(self, amp, pha):
        amp, pha = self._temporal_block(amp, pha)

        return amp, pha

class ChannelStream(nn.Module):
    def __init__(self, 
                 d_model: int, n_heads: int, kernel_sizes: List[int], dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = 'gelu',
                 use_mix_attn: bool = False,
                 use_reverse: bool = True) -> None:
        super(ChannelStream, self).__init__()
        print("[MODEL] Create ChannelStream")
        print(f"\tChannelStream info -> attn_type: {'Mix' if use_mix_attn else 'Cross'} Attention")
        print(f"\tChannelStream info -> reverse_attention: {use_reverse}")
        print(f"\tChannelStream info -> <d_model:{d_model}, n_heads:{n_heads}, kernel_sizes:{kernel_sizes}>")
        print(f"\tChannelStream info -> <dim_feedforward:{dim_feedforward}, dropout:{dropout}, activation:{activation}>")
        self.use_mix_attn = use_mix_attn
        if not use_mix_attn:
            self.pha_ca = Cross_Attn_Block(d_model, n_heads, dropout)
            self.amp_ca = Cross_Attn_Block(d_model, n_heads, dropout)
        else:
            self.mix_attn = Mix_Attn_Block(d_model, n_heads, dropout)

        self.amp_msc = MultiScale_Convolution_Block(d_model, kernel_sizes, dim_feedforward, dropout, activation)
        self.pha_msc = MultiScale_Convolution_Block(d_model, kernel_sizes, dim_feedforward, dropout, activation)
        self.rev_att = Rev_Attn_Block(d_model, dim_feedforward, dropout, activation, use_reverse)
    
    def _channel_block(self, amp, pha):
        if not self.use_mix_attn:
            amp_msc = self.amp_msc(self.amp_ca(amp, pha, pha))
            pha_msc = self.pha_msc(self.pha_ca(pha, amp, amp))
        else:
            amp_ma, pha_ma = self.mix_attn(amp, pha)
            amp_msc = self.amp_msc(amp_ma)
            pha_msc = self.pha_msc(pha_ma)

        amp, pha, rev_amp, rev_pha = self.rev_att(amp_msc, pha_msc, amp, pha)

        return amp, pha, rev_amp, rev_pha
    
    def forward(self, amp, pha):
        amp, pha, rev_amp, rev_pha = self._channel_block(amp, pha)
        
        return amp, pha, rev_amp, rev_pha

class ERCFormerEncoderLayer(nn.Module):
    def __init__(self, 
                 RxTx_num: int,
                 d_model_tem: int, d_model_cha: int,
                 n_heads_tem: int, n_heads_cha: int,
                 dim_feedforward_tem: int, dim_feedforward_cha: int,
                 kernel_sizes: List[int], dropout: float = 0.1, activation: str = 'gelu', norm_first: bool = True,
                 use_mix_attn: bool = False,
                 use_reverse: bool = True) -> None:
        super(ERCFormerEncoderLayer, self).__init__()
        self.norm_first = norm_first
        self.RxTx_num = RxTx_num
        self.amp_sa = Self_Attn_Block(d_model_tem, n_heads_tem, dropout)
        self.pha_sa = Self_Attn_Block(d_model_tem, n_heads_tem, dropout)
        self.temporal_stream = TemporalStream(d_model_tem, n_heads_tem, kernel_sizes, dim_feedforward_tem, dropout, activation, use_mix_attn)
        self.channel_stream = ChannelStream(d_model_cha, n_heads_cha, kernel_sizes, dim_feedforward_cha, dropout, activation, use_mix_attn, use_reverse)
        self.amp_norm = nn.LayerNorm(d_model_tem)
        self.pha_norm = nn.LayerNorm(d_model_tem)

    def forward(self, amp: Tensor, pha: Tensor, return_channel_stream: bool=True, src_mask:Optional[torch.Tensor]=None):
        """
        Args:
            amp (Tensor): Amplitude data, Shape: [batch, frame_num(51), d_model_tem(3072)]
            pha (Tensor): Phase data, Shape: [batch, frame_num(51), d_model_tem(3072)]

        """
        amp_shape, pha_shape = amp.shape, pha.shape
        amp = self.amp_sa(amp, src_mask=src_mask)
        pha = self.pha_sa(pha, src_mask=src_mask)
        amp_cha = amp.clone().view(amp_shape[0], amp_shape[1]*self.RxTx_num, amp_shape[2]//self.RxTx_num).transpose(1, 2)
        pha_cha = pha.clone().view(pha_shape[0], pha_shape[1]*self.RxTx_num, pha_shape[2]//self.RxTx_num).transpose(1, 2)
        amp, pha = self.temporal_stream(amp, pha)
        amp_cha, pha_cha, rev_amp, rev_pha = self.channel_stream(amp_cha, pha_cha)
        amp_cha = amp_cha.transpose(1, 2).reshape(amp_shape)
        pha_cha = pha_cha.transpose(1, 2).reshape(pha_shape)

        if self.norm_first:
            amp = self.amp_norm(amp) + self.amp_norm(amp_cha)
            pha = self.pha_norm(pha) + self.pha_norm(pha_cha)
        else:
            amp = self.amp_norm(amp + amp_cha)
            pha = self.pha_norm(pha + pha_cha)

        if return_channel_stream:
            return amp, pha, rev_amp, rev_pha
        else:
            rev_amp, rev_pha = None, None
            return amp, pha

class ERCFormerEncoder(nn.Module):
    def __init__(self, encoder_layer: ERCFormerEncoderLayer, num_layers: int, return_channel_stream: bool=True) -> None:
        super(ERCFormerEncoder, self).__init__()
        self.encoder_layers = self._make_layers(encoder_layer, num_layers)
        self.return_channel_stream = return_channel_stream

    def _make_layers(self, encoder_layer: ERCFormerEncoderLayer, num_layers):
        return nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
    
    def forward(self, amp: Tensor, pha: Tensor, src_mask:Optional[torch.Tensor]=None):
        """
        Args:
            amp (Tensor): Amplitude data, Shape: [batch, frame_num(51), d_model_tem(3072)]
            pha (Tensor): Phase data, Shape: [batch, frame_num(51), d_model_tem(3072)]

        """
        amp_cha, pha_cha = None, None
        for encoder_layer in self.encoder_layers:
            if not self.return_channel_stream:
                amp, pha = encoder_layer(amp, pha, return_channel_stream=False, src_mask=src_mask)
            else:
                if amp_cha is None:
                    amp, pha, amp_cha, pha_cha = encoder_layer(amp, pha, return_channel_stream=True, src_mask=src_mask)
                else:
                    amp, pha, _, _ = encoder_layer(amp, pha, return_channel_stream=True, src_mask=src_mask)

        if self.return_channel_stream:
            return amp, pha, amp_cha, pha_cha
        else:
            return amp, pha

class ERC_Transformer(nn.Module):
    def __init__(self, 
                 seq_len: int = 51,
                 subcarrier_num: int = 2025,
                 RxTx_num: int = 6,
                 embedding_type: str = 'spatial_temporal',
                 d_model_tem: int = 3072, d_model_cha: int = 306, 
                 n_heads_tem: int = 6, n_heads_cha: int = 6,
                 dim_feedforward_tem: int = 4096, dim_feedforward_cha: int = 1024, 
                 kernel_sizes: List[int] = [1,3,5,7],
                 num_layers: int = 4,
                 dropout: float = 0.1, activation: str = 'gelu', norm_first: bool = True, return_channel_stream: bool = True,
                 gaussian_k: int = 12,
                 use_mix_attn: bool = False,
                 npy_num: int = 3,
                 use_reverse: bool = True) -> None:
        super(ERC_Transformer, self).__init__()
        print("[MODEL] Create ERC_Transformer...")
        encoder_layer = ERCFormerEncoderLayer(RxTx_num, d_model_tem, d_model_cha, n_heads_tem, n_heads_cha,
                                              dim_feedforward_tem, dim_feedforward_cha, 
                                              kernel_sizes, dropout, activation, norm_first, use_mix_attn, use_reverse)
        self.encoder = ERCFormerEncoder(encoder_layer, num_layers, return_channel_stream=return_channel_stream)
        self.return_channel_stream = return_channel_stream
        print("[MODEL] ERC_Transformer created. Number of layers: ", num_layers)

        if embedding_type == 'spatial_temporal':
            self.embedding = Spacial_Temporal_Embedding(d_model_tem, [seq_len, RxTx_num, subcarrier_num], npy_num)
        elif embedding_type == 'gaussian_range':
            self.embedding = Gaussian_Range_Embedding(d_model_tem, [seq_len, RxTx_num, subcarrier_num], gaussian_k, npy_num)

    def forward(self, amp:torch.Tensor, pha:torch.Tensor, src_mask:Optional[torch.Tensor]=None):
        amp, pha = self.embedding(amp, pha)
        if self.return_channel_stream:
            amp, pha, amp_cha, pha_cha = self.encoder(amp, pha, src_mask=src_mask)
            return amp, pha, amp_cha, pha_cha
        else:
            amp, pha = self.encoder(amp, pha, src_mask=src_mask)
            return amp, pha
        
    def infer_mode(self):
        layer: ERCFormerEncoderLayer = None
        for layer in self.encoder.encoder_layers:
            layer.channel_stream.rev_att.infer_mode = True
            layer.channel_stream.rev_att.amp_contrasive_linear = None
            layer.channel_stream.rev_att.pha_contrasive_linear = None
        torch.cuda.empty_cache()
