from torch import nn

import torch
import math


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, head_num, drop_out_rate):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.head_num = head_num
        self.drop_out_rate = drop_out_rate
        
        self.layer_norm_1 = LayerNormalization(self.d_model)
        self.multihead_attention = MultiheadAttention(self.d_model, self.head_num, self.drop_out_rate)
        self.drop_out_1 = nn.Dropout(self.drop_out_rate)

        self.layer_norm_2 = LayerNormalization(self.d_model)
        self.feed_forward = FeedFowardLayer(self.d_model, self.d_ff, self.drop_out_rate)
        self.drop_out_2 = nn.Dropout(self.drop_out_rate)

    def forward(self, x, e_mask):
        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_mask)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(self.feed_forward(x_2)) # (B, L, d_model)

        return x # (B, L, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, head_num, drop_out_rate):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.head_num = head_num
        self.drop_out_rate = drop_out_rate
        
        self.layer_norm_1 = LayerNormalization(self.d_model)
        self.masked_multihead_attention = MultiheadAttention(self.d_model, self.head_num, self.drop_out_rate)
        self.drop_out_1 = nn.Dropout(self.drop_out_rate)

        self.layer_norm_2 = LayerNormalization(self.d_model)
        self.multihead_attention = MultiheadAttention(self.d_model, self.head_num, self.drop_out_rate)
        self.drop_out_2 = nn.Dropout(self.drop_out_rate)

        self.layer_norm_3 = LayerNormalization(self.d_model)
        self.feed_forward = FeedFowardLayer(self.d_model, self.d_ff, self.drop_out_rate)
        self.drop_out_3 = nn.Dropout(self.drop_out_rate)

    def forward(self, x, e_output, e_mask,  d_mask):
        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        ) # (B, L, d_model)
        x_3 = self.layer_norm_3(x) # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3)) # (B, L, d_model)

        return x # (B, L, d_model)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, head_num, drop_out_rate):
        super().__init__()
        self.inf = 1e9
        self.d_model = d_model
        self.head_num = head_num
        self.d_k = d_model // head_num

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into head_num
        q = self.w_q(q).view(input_shape[0], -1, self.head_num, self.d_k) # (B, L, H, d_k)
        k = self.w_k(k).view(input_shape[0], -1, self.head_num, self.d_k) # (B, L, H, d_k)
        v = self.w_v(v).view(input_shape[0], -1, self.head_num, self.d_k) # (B, L, H, d_k)

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
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, H, L, d_k)

        return attn_values


class FeedFowardLayer(nn.Module):
    def __init__(self, d_model, d_ff, drop_out_rate):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.drop_out_rate = drop_out_rate
        
        self.linear_1 = nn.Linear(self.d_model, self.d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.d_ff, self.d_model, bias=True)
        self.dropout = nn.Dropout(self.drop_out_rate)

    def forward(self, x):
        x = self.relu(self.linear_1(x)) # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x) # (B, L, d_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.layer = nn.LayerNorm([self.d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x


class PositionalEncoder(nn.Module):
    def __init__(self, max_len, d_model, device):
        super().__init__()
        self.device = device
        self.max_len = max_len
        self.d_model = d_model
        
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(self.max_len, self.d_model) # (L, d_model)

        # Calculating position encoding values
        for pos in range(self.max_len):
            for i in range(self.d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / self.d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / self.d_model)))

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        self.positional_encoding = pe_matrix.to(self.device).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(self.d_model) # (B, L, d_model)
        x = x + self.positional_encoding # (B, L, d_model)

        return x