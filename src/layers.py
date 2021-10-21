from torch import nn
from torch.nn import functional as F

import torch
import math


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()        
        self.layer_norm_1 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.drop_out_1 = nn.Dropout(dropout)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, dropout)
        self.drop_out_2 = nn.Dropout(dropout)

    def forward(self, x, e_masks):  # x: (B, T, d_model), e_masks: (B, T)
        x_1 = self.layer_norm_1(x) # (B, T, d_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_masks)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(self.feed_forward(x_2)) # (B, L, d_model)

        return x # (B, L, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model)
        self.masked_multihead_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.drop_out_1 = nn.Dropout(dropout)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.drop_out_2 = nn.Dropout(dropout)

        self.layer_norm_3 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, dropout)
        self.drop_out_3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, e_masks, d_masks):  # x: (B, L, d_model), e_outputs: (B, T, d_model), e_masks: (B, T), d_masks: (B, L, L)
        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_masks)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_outputs, e_outputs, mask=e_masks)
        ) # (B, L, d_model)
        x_3 = self.layer_norm_3(x) # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3)) # (B, L, d_model)

        return x # (B, L, d_model)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.inf = 1e9
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, H, d_k)
        k = self.w_k(k).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, H, d_k)
        v = self.w_v(v).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, H, d_k)

        # For convenience, convert all tensors in size (B, H, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask) # (B, H, L, d_k)
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, self.d_model) # (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, H, L, L)
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = F.softmax(attn_scores, dim=-1)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, H, L, d_k)

        return attn_values


class FeedFowardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear_1(x)) # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x) # (B, L, d_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=eps)

    def forward(self, x):
        x = self.layer(x)

        return x
