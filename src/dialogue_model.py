from layers import *
from torch import nn

import torch
import numpy as np
import random


class DialogueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Seed fixing
        np.random.seed(777)
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        random.seed(777)
        
        self.config = config
        
        # Transformer components
        self.embedding = nn.Embedding(self.config['vocab_size'], self.config['d_model'])
        self.positional_embedding = PositionalEncoder(self.config['max_len'], self.config['d_model'], self.config['device'])
        self.encoder = Encoder(self.config['d_model'], self.config['d_ff'], self.config['head_num'], self.config['drop_out_rate'], self.config['layer_num'])
        self.decoder = Decoder(self.config['d_model'], self.config['d_ff'], self.config['head_num'], self.config['drop_out_rate'], self.config['layer_num'])
        
        self.output_linear = nn.Linear(self.config['d_model'], self.config['vocab_size'])
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # Context encoding
        self.linear1 = nn.Linear(self.config['d_model'] + self.config['hidden_size'], self.config['d_mid'])
        self.linear2 = nn.Linear(self.config['d_mid'], self.config['d_model'])
        
        # Context RNN
        self.context_rnn = nn.GRU(
            input_size=self.config['d_model'],
            hidden_size=self.config['hidden_size'],
            num_layers=1,
            batch_first=True,
        )
    
    def init_model(self):            
        # Initialize parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            
    def forward(self, src_input, trg_input, e_mask, d_mask, context):
        # Embeddings
        src_emb = self.embedding(src_input)  # (B, L, d_model)
        src_emb = self.positional_embedding(src_emb)  # (B, L, d_model)
        trg_emb = self.embedding(trg_input)  # (B, L, d_model)
        trg_emb = self.positional_embedding(trg_emb)  # (B, L, d_model)
        
        # Encoding phase
        e_output = self.encoder(src_emb, e_mask)  # (B, L, d_model)
        
        # Encoded input & context combination
        e_output = torch.cat((e_output, context.unsqueeze(1).repeat(1,self.config['max_len'],1)), dim=-1)  # (B, L, d_model+d_h)
        e_output = self.linear1(e_output)  # (B, L, d_mid)
        e_output = self.linear2(e_output)  # (B, L, d_model)
        
        # Decoding phase
        d_output = self.decoder(trg_emb, e_output, e_mask, d_mask)  # (B, L, d_model)
        
        output = self.softmax(self.output_linear(d_output))  # (B, L, vocab_size)
        
        # Context update
        next_context = self.context_update(context, e_output)  # (B, d_model)
        
        return output, next_context  # (B, L, vocab_size), (B, d_model)
        
    
    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != self.config['pad_id']).unsqueeze(1)  # (B, 1, L)
        d_mask = (trg_input != self.config['pad_id']).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, self.config['max_len'], self.config['max_len']], dtype=torch.bool).to(self.config['device'])  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask
    
    def context_update(self, prev_context, e_output):
        cur_context = torch.max(e_output, dim=1).values.unsqueeze(1)  # (B, 1, d_model)
        _, next_context = self.context_rnn(cur_context, prev_context.unsqueeze(0))  # (1, B, d_model)
        next_context = next_context.squeeze(0) # (B, d_model)
        
        return next_context

    
class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, head_num, drop_out_rate, layer_num):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.head_num = head_num
        self.drop_out_rate = drop_out_rate
        self.layer_num = layer_num
        
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_ff, self.head_num, self.drop_out_rate) for i in range(self.layer_num)])
        self.layer_norm = LayerNormalization(self.d_model)

    def forward(self, x, e_mask):
        for i in range(self.layer_num):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, head_num, drop_out_rate, layer_num):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.head_num = head_num
        self.drop_out_rate = drop_out_rate
        self.layer_num = layer_num
        
        self.layers = nn.ModuleList([DecoderLayer(self.d_model, self.d_ff, self.head_num, self.drop_out_rate) for i in range(self.layer_num)])
        self.layer_norm = LayerNormalization(self.d_model)

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(self.layer_num):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)    
