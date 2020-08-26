from dialogue_model import *
from custom_data import *
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch
import os, sys
import numpy as np
import argparse
import sentencepiece as spm


class Manager():
    def __init__(self, mode, ckpt_name=None):
        print("Setting training configuration...")
        self.config = {
            'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            'learning_rate': 0.0001,
            'batch_size': 5,
            'epoch_num': 10,
            'max_len': 256,
            'head_num': 8,
            'layer_num': 6,
            'd_model': 512,
            'd_ff': 2048,
            'drop_out_rate': 0.1,
            'max_turn': 35,
            'nucleus_p': 0.95,
            'ckpt_dir': 'saved_models',
            'data_dir': 'data',
            'train_name': 'train',
            'valid_name': 'validation',
            'processed_dir': 'processed',
            'sp_dir': 'trained_sp',
            'sp_prefix': 'sp',
            'pad_id': 0,
            'bos_id': 1,
            'eos_id': 2,
            'unk_id': 3,
            'dialogue_split_line': '[END OF DIALOGUE]',
            'end_command': 'abort!'
        }
        
        # Sentencepiece tokenizer & vocab
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(f"{self.config['sp_dir']}/{self.config['sp_prefix']}.model")
        with open(f"{self.config['sp_dir']}/{self.config['sp_prefix']}.vocab", 'r') as f:
            lines = f.readlines()
        self.config['vocab_size'] = len(lines)
        
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
            train_set = CustomDataset('train', self.config)
            valid_set = CustomDataset('valid', self.config)
            self.train_loader = DataLoader(train_set, shuffle=True, batch_size=self.config['batch_size'])
            self.valid_loader = DataLoader(valid_set, shuffle=True, batch_size=self.config['batch_size'])
        elif mode == 'test':
            print("Loading the trained model...")
            checkpoint = torch.load(f"{self.config['ckpt_dir']}/{ckpt_name}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
              
        print("Setting finished.")
              
    def train(self):
        print("Training starts.")
              
        for epoch in range(1, self.config['epoch_num']+1):
            self.model.train()
            
            train_losses = []  
            for i, batch in enumerate(tqdm(self.train_loader)):
                src_inputs, trg_inputs, trg_outputs = batch[:, :, 0], batch[:, :, 1], batch[:, :, 2]  # (B, T, L)
              
                dialogue_losses = []
                for t in range(self.config['max_turn']):
                    if t == 0:
                        context = torch.zeros(src_inputs.shape[0], self.config['d_model']).to(self.config['device'])
                      
                    if t < self.config['max_turn']-1:
                        src_input, trg_input, trg_output = \
                            src_inputs[:, t].to(self.config['device']), \
                            trg_inputs[:, t].to(self.config['device']), \
                            trg_outputs[: ,t].to(self.config['device'])  # (B, L)
                        e_mask, d_mask = self.model.make_mask(src_input, trg_input)  # (B, 1, L), (B, L, L)
                        
                        output, context = self.model(src_input, trg_input, e_mask, d_mask, context)  # (B, L, vocab_size), (B, d_h)
                        
                        self.optim.zero_grad()
              
                        loss = self.criterion(
                            output.view(-1, self.config['vocab_size']),
                            trg_output.contiguous().view(output.shape[0] * output.shape[1])
                        )
                  
                        loss.backward(retain_graph=True)
                        self.optim.step()
              
                        dialogue_losses.append(loss.item())
                
                        del src_input, trg_input, trg_output, e_mask, d_mask
                        torch.cuda.empty_cache()
                
                train_losses += dialogue_losses
              
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
                src_inputs, trg_inputs, trg_outputs = batch[:, :, 0], batch[:, :, 1], batch[:, :, 2]  # (B, T, L)
              
                dialogue_losses = []
                for t in range(self.config['max_turn']):
                    if t == 0:
                        context = torch.zeros(src_inputs.shape[0], self.config['d_model']).to(self.config['device'])
                      
                    if t < self.config['max_turn']-1:
                        src_input, trg_input, trg_output = \
                            src_inputs[:, t].to(self.config['device']), \
                            trg_inputs[:, t].to(self.config['device']), \
                            trg_outputs[: ,t].to(self.config['device'])  # (B, L)
                        e_mask, d_mask = self.model.make_mask(src_input, trg_input)  # (B, 1, L), (B, L, L)
                        
                        output, context = self.model(src_input, trg_input, e_mask, d_mask, context)  # (B, L, vocab_size), (B, d_h)
              
                        loss = self.criterion(
                            output.view(-1, self.config['vocab_size']),
                            trg_output.contiguous().view(output.shape[0] * output.shape[1])
                        )
              
                        dialogue_losses.append(loss.item())
                
                        del src_input, trg_input, trg_output, e_mask, d_mask
                        torch.cuda.empty_cache()
                    
                valid_losses += dialogue_losses
              
        mean_valid_loss = np.mean(valid_losses)
              
        return mean_valid_loss
        
              
    def test(self):
        pass

    def nucleus_sampling(self, x, context):
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
