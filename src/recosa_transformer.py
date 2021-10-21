from torch import nn
from layers import *

import torch


class ReCoSaTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        d_emb = args.d_model - args.d_pos
        self.word_embedding = nn.Embedding(args.vocab_size, d_emb)
        self.pos_embedding = nn.Embedding(args.trg_max_len, args.d_pos)
        
        # Word Level GRU components
        self.gru = nn.GRU(
            input_size=d_emb,
            hidden_size=d_emb,
            num_layers=args.num_gru_layers,
            dropout=(0.0 if args.num_gru_layers == 1 else args.gru_dropout),
            batch_first=True,
        )
        
        # Encoder & Decoder
        self.encoder = Encoder(
            args.d_model, 
            args.d_ff, 
            args.num_heads, 
            args.dropout, 
            args.num_encoder_layers,
        )
        self.decoder = Decoder(
            args.d_model, 
            args.d_ff, 
            args.num_heads, 
            args.dropout, 
            args.num_decoder_layers,
        )
        
        self.output_linear = nn.Linear(args.d_model, args.vocab_size)
    
    def init_model(self):            
        # Initialize parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            
    def forward(self, src_inputs, trg_inputs, src_poses, trg_poses, e_masks, d_masks):  
        # src_inputs: (B, T, S_L), trg_inputs: (B, T_L), src_poses: (B, T), trg_poses: (B, T_L), e_masks: (B, T), d_masks: (B, T_L, T_L)
        # Embeddings & Masking 
        src_embs = self.src_embedding(src_inputs, src_poses)  # (B, T, d_model)
        trg_embs = self.trg_embedding(trg_inputs, trg_poses)  # (B, T_L, d_model)
        
        # Encoding phase
        e_outputs = self.encoder(src_embs, e_masks)  # (B, T, d_model)
        
        # Decoding phase
        d_outputs = self.decoder(trg_embs, e_outputs, e_masks, d_masks)  # (B, L, d_model)
        
        return self.output_linear(d_outputs)  # (B, L, vocab_size)
    
    def src_embedding(self, src_inputs, src_poses):  # src_inputs: (B, T, S_L), src_poses: (B, T)
        src_embs = self.word_embedding(src_inputs)  # (B, T, L, d_emb)
        max_len, d_emb = src_embs.shape[2], src_embs.shape[3]
        last_hiddens = self.gru(src_embs.view(-1, max_len, d_emb))[1][-1]  # (B*T, d_emb)
        
        batch_size = src_embs.shape[0]
        src_embs = last_hiddens.view(batch_size, -1, d_emb)  # (B, T, d_emb)
        pos_embs = self.pos_embedding(src_poses)  # (B, T, d_pos)
        src_embs = torch.cat((src_embs, pos_embs), dim=-1)  # (B, T, d_model)
        
        return src_embs  # (B, T, d_model)
    
    def trg_embedding(self, trg_inputs, trg_poses):  # trg_inputs: (B, T_L), trg_poses: (B, T_L)
        trg_embs = self.word_embedding(trg_inputs)  # (B, T_L, d_emb)
        pos_embs = self.pos_embedding(trg_poses)  # (B, T_L, d_pos)
        trg_embs = torch.cat((trg_embs, pos_embs), dim=-1)  # (B, T_L, d_model)
        
        return trg_embs  # (B, T_L, d_model)

    
class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, num_heads, dropout) for i in range(num_layers)])
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_masks):  # x: (B, T, d_model), e_masks: (B, T)
        for i in range(self.num_layers):
            x = self.layers[i](x, e_masks)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, num_heads, dropout) for i in range(num_layers)])
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_outputs, e_masks, d_masks):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_outputs, e_masks, d_masks)

        return self.layer_norm(x)    
