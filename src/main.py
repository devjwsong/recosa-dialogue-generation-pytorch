from transformers import *
from dialogue_model import *
from data_process import *
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch
import os, sys


class Manager():
    def __init__(self, mode, ckpt_name=None):
        gpt2_config = GPT2Config()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        print("Setting training configuration..."")
        self.config = {
            'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            'max_len': gpt2_config['n_positions'],
            'hidden_size': gpt2_config['n_embd'],
            'feed_foward_size': 1024,
            'max_turn': 35,
            'rnn_layer_num': 2,
            'batch_size': 4,
            'learning_rate': 0.0001,
            'epoch_nums': 10,
            'nucleus_p': 0.95,
            'ckpt_dir': 'saved_models',
            'pad_id': self.tokenizer._convert_token_to_id['Ġ'],
            'inf': 1e6
        }
        
        # Load model & optimizer      
        print("Loading the model and optimizer...")
        self.model = DialogueModel(self.config)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.self.best_loss = sys.float_info.max
        
        if ckpt_name is not None:
            assert os.path.exists(f"{ckpt_dir}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

            print("Loading checkpoint...")
            checkpoint = torch.load(f"{ckpt_dir}/{ckpt_name}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.best_loss = checkpoint['loss']
        else:
            print("Initializing the model...")
            self.model.init_model()
              
        if mode=='train':
            # Load loss function
            print("Loading loss function...")
            self.criterion = nn.NLLLoss()
            
            # Load train & valid dataset
            print("Loading train & valid data...")
            train_set = CustomDataset(data_type='train', self.config['max_turn'], self.config['max_len'], self.config['pad_id'])
            valid_set = CustomDataset(data_type='valid', self.config['max_turn'], self.config['max_len'], self.config['pad_id'])
            self.train_loader = DataLoader(train_set, shuffle=True, batch_size=self.config['batch_size'])
            self.valid_loader = DataLoader(valid_set, shuffle=True, batch_size=self.config['batch_size'])
              
        print("Setting finished.")
              
    def train(self):
        pass
              
    def test(self):
        pass
              
    def nucleus_sampling(self):
        pass


if __name__=='__main__':
    pass
