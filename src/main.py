from transformers import *
from dialogue_model import *
from data_process import *
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch
import os, sys
import numpy as np
import argparse


class Manager():
    def __init__(self, mode, ckpt_name=None):
        gpt2_config = GPT2Config().from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        print("Setting training configuration...")
        self.config = {
            'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            'max_len': gpt2_config.n_positions,
            'hidden_size': gpt2_config.n_embd,
            'vocab_size': gpt2_config.vocab_size,
            'feed_forward_size': 1024,
            'max_turn': 35,
            'batch_size': 2,
            'learning_rate': 0.0001,
            'epoch_nums': 10,
            'nucleus_p': 0.95,
            'ckpt_dir': 'saved_models',
            'pad_id': self.tokenizer._convert_token_to_id('Ġ'),
            'inf': 1e6,
            'end_command': 'abort!'
        }
        
        # Load model & optimizer      
        print("Loading the model and optimizer...")
        self.model = DialogueModel(self.config).to(self.config['device'])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.best_loss = sys.float_info.max
        
        if not os.path.exists(self.config['ckpt_dir']):
            os.mkdir(self.config['ckpt_dir'])
        
        if ckpt_name is not None:
            assert os.path.exists(f"{ckpt_dir}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

            print("Loading checkpoint...")
            checkpoint = torch.load(f"{self.config['ckpt_dir']}/{ckpt_name}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.best_loss = checkpoint['loss']
        else:
            print("Initializing the model...")
            self.model.init_model()
              
        if mode == 'train':
            # Load loss function
            print("Loading loss function...")
            self.criterion = nn.NLLLoss()
            
            # Load train & valid dataset
            print("Loading train & valid data...")
            train_set = CustomDataset('train', self.config['max_turn'], self.config['max_len'], self.config['pad_id'])
            valid_set = CustomDataset('valid', self.config['max_turn'], self.config['max_len'], self.config['pad_id'])
            self.train_loader = DataLoader(train_set, shuffle=True, batch_size=self.config['batch_size'])
            self.valid_loader = DataLoader(valid_set, shuffle=True, batch_size=self.config['batch_size'])
        elif mode == 'test':
            print("Loading the trained model...")
            checkpoint = torch.load(f"{self.config['ckpt_dir']}/{ckpt_name}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
              
        print("Setting finished.")
              
    def train(self):
        print("Training starts.")
              
        for epoch in range(1, self.config['epoch_nums']+1):
            self.model.train()
            
            train_losses = []  
            for i, batch in enumerate(tqdm(self.train_loader)):
                turn_num, dialogue = batch  # (B), (B, T, L)
                turn_num, dialogue = turn_num.to(self.config['device']), dialogue.to(self.config['device'])
              
                dialogue_losses = []
                for t in range(self.config['max_turn']):
                    if t == 0:
                        context = torch.zeros(dialogue.shape[0], self.config['hidden_size']).to(self.config['device'])
                      
                    if t < self.config['max_turn']-1:
                        output, next_context = self.model(dialogue[:, t], context)  # (B, L, vocab_size), (B, d_h)
                        context = next_context
                        
                        self.optim.zero_grad()
              
                        loss = self.criterion(
                            output.view(-1, self.config['vocab_size']),
                            dialogue[:, t+1].contiguous().view(output.shape[0] * output.shape[1])
                        )
                  
                        loss.backward(retain_graph=True)
                        self.optim.step()
              
                        dialogue_losses.append(loss.item())
                
                train_losses += dialogue_losses
              
                del turn_num, dialogue
                torch.cuda.empty_cache()
              
            mean_train_loss = np.mean(train_losses)
            print(f"#################### Epoch: {epoch} ####################")
            print(f"Train loss: {mean_train_loss}")
            
            valid_loss = self.validation()
              
            if valid_loss < self.best_loss:
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'loss': mean_train_loss
                }
              
                torch.save(state_dict, f"{self.config['ckpt_dir']}/best_ckpt.tar")
                print(f"***** Current best checkpoint is saved. *****")
                self.best_loss = valid_loss
              
            print(f"Best valid loss: {self.best_loss}")
            print(f"Valid loss: {valid_loss}")
              
        print("Training finished!")
    
    def validation(self):
        print("Validation processing...")
        self.model.eval()
              
        valid_losses = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                turn_num, dialogue = batch  # (B), (B, T, L)
                turn_num, dialogue = turn_num.to(self.config['device']), dialogue.to(self.config['device'])
              
                dialogue_losses = []
                for t in range(self.config['max_turn']):
                    if t == 0:
                        context = torch.zeros(dialogue.shape[0], self.config['hidden_size']).to(self.config['device'])
                      
                    if t < self.config['max_turn']-1:
                        output, next_context = self.model(dialogue[:, t], context)  # (B, L, vocab_size), (B, d_h)
                        context = next_context
              
                        loss = self.criterion(
                            output.view(-1, self.config['vocab_size']),
                            dialogue[:, t+1].contiguous().view(output.shape[0] * output.shape[1])
                        )
              
                        dialogue_losses.append(loss.item())
              
                valid_losses += dialogue_losses
                
                del turn_num, dialogue
                torch.cuda.empty_cache()
              
        mean_valid_loss = np.mean(valid_losses)
              
        return mean_valid_loss
        
              
    def test(self):
        print("Dialogue start!")
        self.model.eval()
        
        context = None
        print("Please type your input to start the conversation.")
        print(f"If you want to finish this conversation, please type {self.config['end_command']}.")
        while True:
            input_sentence = input()
            
            if input_sentence == self.config['end_command']:
                break
            else:
                tokens = self.tokenizer.tokenize(input_sentence)
                ids = [self.tokenizer._convert_token_to_id(token) for token in tokens]
                
                if len(ids) <= self.config['max_len']:
                    left = self.config['max_len'] - len(ids)
                    ids += [self.config['pad_id']] * left
                else:
                    ids = ids[:self.config['max_len']]
                    
                x = torch.LongTensor(ids).unsqueeze(0)  # (1, L)
                
                if context is None:
                    context = torch.zeros(dialogue.shape[0], self.config['hidden_size']).to(self.config['device'])
                    
                
                
                
                
        print("Dialogue finished. Thank you.")

    def nucleus_sampling(self, x, context):
        for pos in range(self.config['max_len']):
            pass
        
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help="train or test?")
    parser.add_argument('--ckpt_name', required=False, help="best checkpoint file")
              
    args = parser.parse_args()
              
    if args.mode == 'train':
        if args.ckpt_name is not None:
            manager = Manager(args.mode, ckpt_name=args.ckpt_name)
        else:
            manager = Manager(args.mode)
              
        manager.train()
        
    elif args.mode == 'test':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint."
        
        manager = Manager(args.mode, ckpt_name=args.ckpt_name)
        
        manager.test()
