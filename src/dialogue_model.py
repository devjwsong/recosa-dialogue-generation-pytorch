from transformers import *
from torch import nn

import torch
import numpy as np
import random


class DialogueModel(nn.Module):
    def __init__(self, config):
        
        # Seed fixing
        np.random.seed(777)
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        random.seed(777)
        
        gpt2 = GPT2Model.from_pretrained('gpt2')
        self.config = config
        
        # GPT2 components
        self.embedding = gpt2.wte
        self.positional_embedding = gpt2.wpe
        self.drop = gpt2.drop
        self.decoder = gpt2.h
        self.ln_f = gpt2.ln_f
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # Context encoding
        self.linear1 = nn.Linear(2*self.config['hidden_size'], self.config['feed_forward_size'])
        self.linear2 = nn.Linear(self.config['feed_forward_size'], self.config['hidden_size'])
        
        # Context RNN
        self.context_rnn = nn.GRU(
            input_size=self.config['hidden_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['rnn_layer_num'],
            batch_first=True,
            drop_out=0.2
        )
    
    def init_model(self):
        # Freeze word embedding layer
        for param in self.embedding.parameters():
            param.requires_grad = False
            
        # Initialize fully connected layers
        nn.init.xavier_uniform(self.linear1.weight)
        nn.init.xavier_uniform(self.linear2.weight)
        
        # Initialize GRU
        for param in self.context_rnn.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            
    def forward(self, x, context):
        x_embedded = self.embedding(x)  # (B, L, d_h)
        
        position_ids = torch.arange(0, x.shape[-1], dtype=torch.long).to(self.config['device'])
        p_embedded = self.positional_embedding(position_ids)  # (B, L, d_h)
        
        hidden_states = x_embedded + p_embedded
        
        # Utterance & Context combination
        max_len = hidden_states.shape[1]
        hidden_states = torch.cat((hidden_states, context.unsqueeze(1).repeat(1,max_len,1)), dim=-1)  # (B, L, 2d_h)
        hidden_states = self.linear1(hidden_states)  # (B, L, d_ff)
        hidden_states = self.linear2(hidden_states)  # (B, L, d_h)
        
        hidden_states = self.drop(hidden_states)  # (B, L, d_h)
        
        attention_mask = self.make_mask(x)  # (B, 1, 1, L)
        
        # Decoding phase
        for i, layer in enumerate(self.decoder):
            outputs = layer(
                hidden_states,
                layer_past=None,
                attention_mask=attention_mask,
                head_mask=None
                use_cache=False
            )
            
            hidden_states, _ = outputs[:2]
            
        hidden_states = self.ln_f(hidden_states)  # (B, L, d_h)
        
        # Context update
        current_context = torch.max(x_embedded + p_embedded, dim=1).unsqueeze(1)  # (B, 1, d_h)
        prev_context = context.unsqueeze(0)  # (1, B, d_h)
        _, next_context = self.context_rnn(current_context, prev_context)
        next_context = next_context.squeeze(0)  # (B, d_h)
        
        return self.softmax(hidden_states), next_context  # (B, L, d_h), (B, d_h)
        
        
    
    def make_mask(self, input_tensor):
        mask = (input_tensor != config['pad_id']).float()  # (B, L)
        mask = mask.view(input_tensor.shape[0], -1)
        mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        mask = mask.to(dtype=next(self.decoder.parameters()).dtype)
        
        return (1.0 - mask) * self.config['inf']  # (B, 1, 1, L)
