from torch import nn
from layers import *

import torch
import math
import numpy as np
import random


class GRUTransformer(nn.Module):
    def __init__(self, config, embedding=None):
        super().__init__()
        
        # Seed fixing
        np.random.seed(777)
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        random.seed(777)
        
        self.config = config
        self.use_gpt = False
        if embedding is not None:
            self.use_gpt = True
        
        # Transformer components
        if self.use_gpt:
            self.embedding = embedding  # GPT2 word embedding layer
            self.embedding_linear = nn.Linear(self.embedding.embedding_dim, self.config['d_model'])
        else:
            self.embedding = nn.Embedding(self.config['vocab_size'], self.config['d_model'])
            
        self.positional_embedding = PositionalEncoder(self.config['max_len'], self.config['d_model'], self.config['device'])
        self.encoder = Encoder(self.config['d_model'], self.config['d_ff'], self.config['num_heads'], self.config['dropout'], self.config['encoder_num_layers'])
        self.decoder = Decoder(self.config['d_model'], self.config['d_ff'], self.config['num_heads'], self.config['dropout'], self.config['decoder_num_layers'])
        
        self.output_linear = nn.Linear(self.config['d_model'], self.config['vocab_size'])
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # Context encoding
        self.linear1 = nn.Linear(self.config['d_model'] + self.config['hidden_size'], self.config['d_mid'])
        self.linear2 = nn.Linear(self.config['d_mid'], self.config['d_model'])
        
        # Context RNN
        self.context_rnn = nn.GRU(
            input_size=self.config['d_model'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['gru_num_layers'],
            dropout=self.config['gru_dropout'],
            batch_first=True,
        )
    
    def init_model(self):            
        # Initialize parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            
    def forward(self, src_input, trg_input, context):
        # Embeddings & Masking
        src_emb = self.embed(src_input)  # (B, L, d_model)
        trg_emb = self.embed(trg_input)  # (B, L, d_model)
        e_mask = self.make_encoder_mask(src_input)  # (B, 1, L)
        d_mask = self.make_decoder_mask(trg_input)  # (B, L, L)
        
        # Encoding phase
        e_output = self.encoder(src_emb, e_mask)  # (B, L, d_model)
        
        # Context update
        next_context = self.context_update(context, e_output)  # (B, d_model)
        
        # Decoding phase
        e_output = self.combine_context(e_output, context)  # (B, L, d_model)
        d_output = self.decoder(trg_emb, e_output, e_mask, d_mask)  # (B, L, d_model)
        
        output = self.softmax(self.output_linear(d_output))  # (B, L, vocab_size)
        
        del e_mask, d_mask
        
        return output, next_context  # (B, L, vocab_size), (B, d_model)
        
    def make_encoder_mask(self, src_input):
        e_mask = (src_input != self.config['pad_id']).unsqueeze(1)  # (B, 1, L)
        
        return e_mask
    
    def make_decoder_mask(self, trg_input):
        d_mask = (trg_input != self.config['pad_id']).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, self.config['max_len'], self.config['max_len']], dtype=torch.bool).to(self.config['device'])  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false
        
        return d_mask
    
    def embed(self, input_x):
        x_emb = self.embedding(input_x)  # (B, L, d_model) or (B, L, e_dim)
        if self.use_gpt:
            x_emb = self.embedding_linear(x_emb)  # (B, L, d_model)
        x_emb = self.positional_embedding(x_emb, cal='add')  # (B, L, d_model)
    
        return x_emb
    
    def combine_context(self, e_output, context):
        e_output = torch.cat((e_output, context.unsqueeze(1).repeat(1,self.config['max_len'],1)), dim=-1)  # (B, L, d_model+d_h)
        e_output = self.linear1(e_output)  # (B, L, d_mid)
        e_output = self.linear2(e_output)  # (B, L, d_model)
        
        return e_output
    
    def context_update(self, prev_context, e_output):
        cur_context = torch.max(e_output, dim=1).values.unsqueeze(1)  # (B, 1, d_model)
        _, next_context = self.context_rnn(cur_context, prev_context.unsqueeze(0))  # (1, B, d_model)
        next_context = next_context.squeeze(0) # (B, d_model)
        
        return next_context

    
class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_ff, self.num_heads, self.dropout) for i in range(self.num_layers)])
        self.layer_norm = LayerNormalization(self.d_model)

    def forward(self, x, e_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([DecoderLayer(self.d_model, self.d_ff, self.num_heads, self.dropout) for i in range(self.num_layers)])
        self.layer_norm = LayerNormalization(self.d_model)

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)    
