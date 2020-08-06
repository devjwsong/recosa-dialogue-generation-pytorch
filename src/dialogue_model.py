from transformers import *
from torch import nn

import torch


class DialogueModel(nn.Module):
    def __init__(self, config):
        gpt2 = GPT2Model.from_pretrained('gpt2')
        
        # GPT2 components
        self.embedding = gpt2.wte
        self.positional_embedding = gpt2.wpe
        self.drop = gpt2.drop
        self.decoder = gpt2.h
        self.ln_f = gpt2.ln_f
        
        # Context encoding
        self.linear1 = nn.Linear(2*config['hidden_size'], config['feed_forward_size'])
        self.linear2 = nn.Linear(config['feed_forward_size'], config['hidden_size'])
    
    def init_model(self):
        # Freeze word embedding layer
        for param in self.embedding.parameters():
            param.requires_grad = False
            
        # Initialize fully connected layers
        nn.init.xavier_uniform(self.linear1.weight)
        nn.init.xavier_uniform(self.linear2.weight)
            
    def forward(self, x, context):
        pass
    
    def make_mask(self, input_tensor):
        return (input_tensor != config['pad_id']).long()  # (B, L)
