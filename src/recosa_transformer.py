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
        self.turn_pembedding = PositionalEncoder(self.config['max_turn'], self.config['d_model'], self.config['device'])
        
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
            
    def forward(self, src_input, trg_input, hists, num_turn):
        # Embeddings & Masking
        src_emb, hists = self.src_embed(src_input, hists, num_turn)  # (B, T, 2*d_model), (T, B, d_model)
        trg_emb = self.trg_embed(trg_input)  # (B, L, 2*d_model)
        e_mask = self.make_encoder_mask(src_input, num_turn)  # (B, 1, T)
        d_mask = self.make_decoder_mask(trg_input)  # (B, L, L)
        
        # Encoding phase
        e_output = self.encoder(src_emb, e_mask)  # (B, L, 2*d_model)
        
        # Decoding phase
        d_output = self.decoder(trg_emb, e_output, e_mask, d_mask)  # (B, L, 2*d_model)
        
        output = self.softmax(self.output_linear(d_output))  # (B, L, vocab_size)
        
        del e_mask, d_mask
        
        return output, hists  # (B, L, vocab_size), (T, B, d_model)
        
    def make_encoder_mask(self, src_input, num_turn):
        e_mask = torch.BoolTensor([1 for i in range(num_turn+1)] + [0 for i in range(self.config['max_turn']-num_turn-1)]).to(self.config['device'])
        e_mask = e_mask.unsqueeze(0).repeat(src_input.shape[0], 1).unsqueeze(1)  # (B, 1, T)
        
        return e_mask
    
    def make_decoder_mask(self, trg_input):
        d_mask = (trg_input != self.config['pad_id']).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, self.config['max_len'], self.config['max_len']], dtype=torch.bool).to(self.config['device'])  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false
        
        return d_mask
    
    def src_embed(self, src_input, hists, num_turn):
        src_emb = self.embedding(src_input)  # (B, L, d_model) or (B, L, e_dim)
        if self.use_gpt:
            src_emb = self.embedding_linear(src_emb)  # (B, L, d_model)
        last_hist = self.word_level_rnn(src_emb)[1][-1]  # (B, d_model)

        hists[num_turn] = last_hist  # (T, B, d_model)
        src_emb = hists.transpose(0, 1)  # (B, T, d_model)
        src_emb = self.turn_pembedding(src_emb, cal='concat')  # (B, T, 2*d_model)
        
        return src_emb, hists  # (B, T, 2*d_model), (T, B, d_model)
    
    def trg_embed(self, trg_input):
        trg_emb = self.embedding(trg_input)  # (B, L, d_model) or (B, L, e_dim)
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
