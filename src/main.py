from dialogue_model import *
from custom_data import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torch
import os, sys
import numpy as np
import argparse
import sentencepiece as spm
import time


class Manager():
    def __init__(self, mode, ckpt_name=None):
        print("Setting the configurations...")
        self.config = {
            'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            'learning_rate': 0.00001,
            'batch_size': 5,
            'epoch_num': 10,
            'max_len': 256,
            'head_num': 8,
            'layer_num': 6,
            'd_model': 512,
            'hidden_size': 128,
            'd_mid': 768,
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
            assert os.path.exists(f"{self.config['ckpt_dir']}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

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
              
        print("Setting finished.")
              
    def train(self):
        print("Training starts.")
              
        for epoch in range(1, self.config['epoch_num']+1):
            self.model.train()
            
            print(f"#################### Epoch: {epoch} ####################")
            train_losses = []  
            for i, batch in enumerate(tqdm(self.train_loader)):
                src_inputs, trg_inputs, trg_outputs = batch[:, :, 0], batch[:, :, 1], batch[:, :, 2]  # (B, T, L)
              
                dialogue_losses = []
                for t in range(self.config['max_turn']):
                    if t == 0:
                        context = torch.zeros(src_inputs.shape[0], self.config['hidden_size']).to(self.config['device'])
                      
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
                        context = torch.zeros(src_inputs.shape[0], self.config['hidden_size']).to(self.config['device'])
                      
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
        print("Let's start!")
        print(f"If you want to quit the converstation, please type \"{self.config['end_command']}\".")
        self.model.eval()
        
        with torch.no_grad():
            utter = None
            context = None
            for t in range(self.config['max_turn']):
                if t % 2 == 0:
                    utter = input("You: ")
                    
                if utter == self.config['end_command']:
                    print("Bot: Good bye.")
                    break

                tokens = self.tokenizer.EncodeAsIds(utter)
                if len(tokens) < self.config['max_len']:
                    src_input = tokens + [self.config['eos_id']]
                    src_input += [self.config['pad_id']] * (self.config['max_len'] - len(src_input))
                else:
                    src_input = src_input[:self.config['max_len']]
                    src_input[-1] = self.config['eos_id']

                src_input = torch.LongTensor(src_input).unsqueeze(0).to(self.config['device'])  # (B, L)
                e_mask = (src_input != self.config['pad_id']).unsqueeze(1)  # (B, 1, L)

                if t == 0:
                    context = torch.zeros(src_input.shape[0], self.config['hidden_size']).to(self.config['device'])

                src_emb = self.model.embedding(src_input)  # (B, L, d_model)
                src_emb = self.model.positional_embedding(src_emb)  # (B, L, d_model)

                e_output = self.model.encoder(src_emb, e_mask)  # (B, L, d_model)
                e_output = torch.cat((e_output, context.unsqueeze(1).repeat(1,self.config['max_len'],1)), dim=-1)  # (B, L, d_model + d_h)
                e_output = self.model.linear1(e_output)  # (B, L, d_mid)
                e_output = self.model.linear2(e_output)  # (B, L, d_model)

                context = self.model.context_update(context, e_output)

                if t % 2 == 0:
                    output_ids = self.nucleus_sampling(e_output, e_mask)  # (L) list
                    utter = self.tokenizer.DecodeIds(output_ids)

                    print(f"Bot: {utter}")

                if t == self.config['max_turn']-1:
                    print("This is the last turn.")

    def nucleus_sampling(self, e_output, e_mask):
        trg_input = [self.config['bos_id']]
        trg_input += [self.config['pad_id']] * (self.config['max_len']-len(trg_input))
        trg_input = torch.LongTensor(trg_input).unsqueeze(0).to(self.config['device'])  # (B, L)
        
        output_ids = []
        
        for pos in range(self.config['max_len']):
            d_mask = (trg_input != self.config['pad_id']).unsqueeze(1)  # (B, 1, L)
            nopeak_mask = torch.ones([1, self.config['max_len'], self.config['max_len']], dtype=torch.bool).to(self.config['device'])  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask  # (B, L, L) padding false
            
            trg_emb = self.model.embedding(trg_input)  # (B, L, d_model)
            trg_emb = self.model.positional_embedding(trg_emb)  # (B, L, d_model)
            d_output = self.model.decoder(trg_emb, e_output, e_mask, d_mask)  # (B, L, d_model)
            
            output = F.softmax(self.model.output_linear(d_output), dim=-1)  # (B, L, vocab_size)
            output = output[:,pos]  # (B, vocab_size)
            
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (B, vocab_size)
            idx_remove = cumsum_probs > self.config['nucleus_p']
            sorted_probs[idx_remove] = 1e-8
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (B, vocab_size)
            
            
            # Random sampling
            seed = int(time.time())
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            probs = torch.zeros(output.shape).to(self.config['device']).scatter_(-1, sorted_idxs, sorted_probs)  # (B, vocab_size)
            idxs = torch.multinomial(probs, 1).squeeze(-1)  # (B)
            
            if pos < self.config['max_len']-1:
                trg_input[:, pos+1] = idxs
            
            output_ids.append(idxs.squeeze(0).item())    
            if idxs.squeeze(0).item() == self.config['eos_id']:
                break
            
            del output, sorted_probs, sorted_idxs, cumsum_probs, idx_remove, probs, idxs
            torch.cuda.empty_cache()
            
        if output_ids[-1]== self.config['eos_id']:
            output_ids = output_ids[:-1]
            
        return output_ids
        

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
