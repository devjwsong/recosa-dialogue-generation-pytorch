from torch import nn
from layers import *

import torch
import math
import numpy as np
import random


class ReCoSaTransformer(nn.Module):
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
        
        # Embedding components
        if self.use_gpt:
            self.embedding = embedding  # GPT2 word embedding layer
            self.embedding_linear = nn.Linear(self.embedding.embedding_dim, self.config['d_model'])
        else:
            self.embedding = nn.Embedding(self.config['vocab_size'], self.config['d_model'])
        self.word_pembedding = PositionalEncoder(self.config['max_len'], self.config['d_model'], self.config['device'])
        self.time_pembedding = PositionalEncoder(self.config['max_time'], self.config['d_model'], self.config['device'])
        
        # Word Level LSTM components
        self.word_level_rnn = nn.GRU(
            input_size=self.config['d_model'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['gru_num_layers'],
            dropout=self.config['gru_dropout'],
            batch_first=True,
        )
        
        # Encoder & Decoder
        self.encoder = Encoder(
            self.config['hidden_size'] + self.config['d_model'], 
            self.config['d_ff'], 
            self.config['num_heads'], 
            self.config['dropout'], 
            self.config['encoder_num_layers']
        )
        self.decoder = Decoder(
            self.config['hidden_size'] + self.config['d_model'], 
            self.config['d_ff'], 
            self.config['num_heads'], 
            self.config['dropout'], 
            self.config['decoder_num_layers']
        )
        
        self.output_linear = nn.Linear(self.config['hidden_size'] + self.config['d_model'], self.config['vocab_size'])
        self.softmax = nn.LogSoftmax(dim=-1)
        
    
    def init_model(self):            
        # Initialize parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            
    def forward(self, src_input, trg_input, e_mask, d_mask):
        # Embeddings & Masking
        src_emb = self.src_embed(src_input)  # (B, T, 2*d_model)
        trg_emb = self.trg_embed(trg_input)  # (B, L, 2*d_model)
        
        # Encoding phase
        e_output = self.encoder(src_emb, e_mask)  # (B, T, 2*d_model)
        
        # Decoding phase
        d_output = self.decoder(trg_emb, e_output, e_mask, d_mask)  # (B, L, 2*d_model)
        
        output = self.softmax(self.output_linear(d_output))  # (B, L, vocab_size)
        
        return output  # (B, L, vocab_size)
    
    def src_embed(self, src_input):
        src_emb = self.embedding(src_input)  # (B, T, L, d_model)
        if self.use_gpt:
            src_emb = self.embedding_linear(src_emb)  # (B, T, L, d_model)
        max_len, d_model = src_emb.shape[2], src_emb.shape[3]
        last_hiddens = self.word_level_rnn(src_emb.view(-1, max_len, d_model))[1][-1]  # (B*T, d_model)

        batch_size = src_emb.shape[0]
        src_emb = last_hiddens.view(batch_size, -1, d_model)  # (B, T, d_model)
        src_emb = self.time_pembedding(src_emb, cal='concat')  # (B, T, 2*d_model)
        
        return src_emb  # (B, T, 2*d_model)
    
    def trg_embed(self, trg_input):
        trg_emb = self.embedding(trg_input)  # (B, L, d_model)
        if self.use_gpt:
            trg_emb = self.embedding_linear(trg_emb)  # (B, L, d_model)
        trg_emb = self.word_pembedding(trg_emb, cal='concat')  # (B, L, 2*d_model)
        
        return trg_emb  # (B, L, 2*d_model)

    
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
