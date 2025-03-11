from typing import Optional
from torch import nn
import torch
import math

def mix_scaled_dot_product_attention(a_qkey: torch.Tensor, b_qkey: torch.Tensor,
                                     a_value: torch.Tensor, b_value: torch.Tensor,
                                     attn_mask: Optional[torch.Tensor] = None,
                                     dropout_p: float = 0.0, scale: Optional[float] = None) -> torch.Tensor:
    a_len, b_len = a_qkey.size(-2), b_qkey.size(-2)
    scale_factor = 1 / math.sqrt(a_qkey.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(a_len, b_len, device=a_qkey.device, dtype=a_qkey.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.clone().masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = a_qkey @ b_qkey.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight_a = torch.softmax(attn_weight, dim=-1)
    attn_weight_b = torch.softmax(attn_weight, dim=-2).transpose(-2, -1)
    attn_weight_a = torch.dropout(attn_weight_a, dropout_p, train=True)
    attn_weight_b = torch.dropout(attn_weight_b, dropout_p, train=True)
    return attn_weight_a @ b_value, attn_weight_b @ a_value

def MixAttentionFroward(a_qkey: torch.Tensor, b_qkey: torch.Tensor, 
                        a_value: torch.Tensor, b_value: torch.Tensor,
                        num_heads: int, head_dim: int, embed_dim: int,
                        linear_a_out: nn.Linear, linear_b_out: nn.Linear,
                        attn_mask=None, dropout_p=0.0, scale=None) -> list[torch.Tensor, torch.Tensor]:
    
    a_length, bsz, E_a = a_qkey.size()
    b_length, bsz, E_b = b_qkey.size()
    a_qkey = a_qkey.view(a_length, bsz*num_heads, head_dim).transpose(0, 1)
    b_qkey = b_qkey.view(b_length, bsz*num_heads, head_dim).transpose(0, 1)
    a_value = a_value.view(a_length, bsz*num_heads, head_dim).transpose(0, 1)
    b_value = b_value.view(b_length, bsz*num_heads, head_dim).transpose(0, 1)

    a_qkey = a_qkey.view(bsz, num_heads, a_length, head_dim)
    b_qkey = b_qkey.view(bsz, num_heads, b_length, head_dim)
    a_value = a_value.view(bsz, num_heads, a_length, head_dim)
    b_value = b_value.view(bsz, num_heads, b_length, head_dim)

    a_out, b_out = mix_scaled_dot_product_attention(a_qkey, b_qkey, a_value, b_value, attn_mask, dropout_p, scale)
    a_out = a_out.permute(2, 0, 1, 3).contiguous().view(bsz * a_length, embed_dim)
    b_out = b_out.permute(2, 0, 1, 3).contiguous().view(bsz * b_length, embed_dim)

    a_out = linear_a_out(a_out).view(a_length, bsz, embed_dim)
    b_out = linear_b_out(b_out).view(b_length, bsz, embed_dim)
    
    return a_out, b_out

class MultiHeadMixAttention(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 batch_first: bool = True,
                 dropout: float = 0.0, bias: bool = True, scale: Optional[float] = None):
        super(MultiHeadMixAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.scale = scale
        self.batch_first = batch_first
        self.linear_a_qkey = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_a_value = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_b_qkey = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_b_value = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_a_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_b_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    
    def _check_mask_and_merge(self, attn_mask, 
                              a_key_padding_mask, b_key_padding_mask) -> Optional[torch.Tensor]:
        merged_mask: Optional[torch.Tensor] = None

        if attn_mask is not None:
            # expand attn_mask to match the atteniton size for SDP -> [n, num_heads, L, S]
            if attn_mask.dim() == 2: # input mask shape: [L, S]
                attn_mask = attn_mask.view(
                    1, 1, self.a_length, self.b_length).expand(
                        self.bsz, self.num_heads, self.a_length, self.b_length)
            elif attn_mask.dim() == 3: # input mask shape: [n*num_heads, L, S]
                attn_mask = attn_mask.view(self.bsz, self.num_heads, self.a_length, self.b_length)
            merged_mask = attn_mask

        if a_key_padding_mask is not None: # input mask shape: [n, L]
            # expand a_key_padding_mask to match the atteniton size for SDP -> [n, num_heads, L, S]
            a_key_padding_mask = a_key_padding_mask.view(
                self.bsz, 1, self.a_length, 1).expand(
                    self.bsz, self.num_heads, self.a_length, self.b_length)
            if merged_mask is None:
                merged_mask = a_key_padding_mask
            else:
                merged_mask = torch.logical_or(merged_mask, a_key_padding_mask)

        if b_key_padding_mask is not None: # input mask shape: [n, S]
            # expand b_key_padding_mask to match the atteniton size for SDP -> [n, num_heads, L, S]
            b_key_padding_mask = b_key_padding_mask.view(
                self.bsz, 1, 1, self.b_length).expand(
                    self.bsz, self.num_heads, self.a_length, self.b_length)
            if merged_mask is None:
                merged_mask = b_key_padding_mask
            else:
                merged_mask = torch.logical_or(merged_mask, b_key_padding_mask)
        return merged_mask
    
    def forward(self, a: torch.Tensor, b: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, 
                a_key_padding_mask: Optional[torch.Tensor] = None,
                b_key_padding_mask: Optional[torch.Tensor] = None):
        assert a.size(-1) == b.size(-1) == self.embed_dim, "input tensor.size(-1) must be equal to embed_dim"
        assert a.dim() == b.dim() == 3, "input tensor must be batched"
        """
        Args:
            a: Input tensor with shape: [n, L, E] for batch_first=True, otherwise [L, n, E].
            b: Input tensor with shape: [n, S, E] for batch_first=True, otherwise [S, n, E].
            attn_mask: Input bool tensor with shape: [L, S] or [n*num_heads, L, S]
            a_key_padding_mask: Input bool tensor with shape: [n, L]
            b_key_padding_mask: Input bool tensor with shape: [n, S]
        Returns:
            a_out: Output tensor with shape: [n, L, E] for batch_first=True, otherwise [L, n, E].
            b_out: Output tensor with shape: [n, S, E] for batch_first=True, otherwise [S, n, E].
        
        Note:
            n is batch size, L is the maximum length of a, S is the maximum length of b, E is embedding dimensions.
            mask:
                A bool tensor where "True" values are positions that should be "masked" with float('-inf').
        """
        # get all size parameters
        if self.batch_first:
            a, b = a.transpose(0, 1), b.transpose(0, 1)
            # a: [L, n, E], b: [S, n, E]
        self.a_length, self.bsz, E_a = a.size()
        self.b_length, self.bsz, E_b = b.size()
        head_dim = self.embed_dim // self.num_heads
        
        # check mask shape and merge attn_mask, a_key_padding_mask, b_key_padding_mask 
        if attn_mask is not None:
            if attn_mask.dim() == 3 and attn_mask.size(-1) != a.size(2)*self.num_heads:
                raise ValueError("The last dimension of attn_mask must be equal to batch_size * num_heads")
            if attn_mask.dim() == 2 and attn_mask.size() != (self.a_length, self.b_length):
                raise ValueError(f"The size of 2 dimensional attn_mask should be \
                                  [{self.a_length, self.b_length}] got {attn_mask.size()}")

        a_qkey: torch.Tensor = self.linear_a_qkey(a)
        a_value: torch.Tensor = self.linear_a_value(a)
        b_qkey: torch.Tensor = self.linear_b_qkey(b)
        b_value: torch.Tensor = self.linear_b_value(b)
        merged_mask = self._check_mask_and_merge(attn_mask, a_key_padding_mask, b_key_padding_mask)
        a_out, b_out = MixAttentionFroward(
            a_qkey, b_qkey, 
            a_value, b_value, 
            self.num_heads, head_dim, self.embed_dim, 
            self.linear_a_out, self.linear_b_out, 
            merged_mask, self.dropout, self.scale)

        if self.batch_first:
            a_out, b_out = a_out.transpose(0, 1), b_out.transpose(0, 1)

        return a_out, b_out
        