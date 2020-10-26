from transformers import *
from recosa_transformer import *
from custom_data import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torch
import os, sys
import numpy as np
import argparse
import time
import copy


class Manager():
    def __init__(self, mode, use_gpt=False, ckpt_name=None):
        print("Setting the configurations...")
        self.config = {
            'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            'learning_rate': 5e-4,
            'batch_size': 26,
            'num_epochs': 20,
            'max_len': 300,
            'num_heads': 8,
            'encoder_num_layers': 6,
            'decoder_num_layers': 6,
            'd_model': 512,
            'd_mid': 768,
            'd_ff': 2048,
            'dropout': 0.1,
            'max_time': 20,
            'nucleus_p': 0.9,
            'ckpt_dir': 'saved_models',
            'data_dir': 'data',
            'train_name': 'train',
            'valid_name': 'validation',
            'dialogue_split_line': '[END OF DIALOGUE]',
            'end_command': 'Abort!',
            'bos': '<bos>',
            'eos': '<eos>',
            'pad': '<pad>',
        }
        self.config['gru_num_layers'] =  2
        self.config['hidden_size'] = self.config['d_model']
        self.config['gru_dropout'] = 0.3
        
        # Tokenizer & Vocab
        print("Loading tokenizer & embedding...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens = {
            'bos_token': self.config['bos'],
            'eos_token': self.config['eos'],
            'pad_token': self.config['pad']
        }
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.config['vocab_size'] = len(vocab)
        self.config['bos_id'] = vocab[self.config['bos']]
        self.config['eos_id'] = vocab[self.config['eos']]
        self.config['pad_id'] = vocab[self.config['pad']]
        
        embedding = None
        if use_gpt:
            gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
            num_ori_tokens = gpt2.transformer.wte.num_embeddings
            gpt2.resize_token_embeddings(num_ori_tokens + num_new_tokens)
            embedding = gpt2.transformer.wte
        
        # Load model    
        print("Loading the model...")
        self.model = ReCoSaTransformer(self.config, embedding=embedding).to(self.config['device'])
            
        if mode == 'train':
            # Load loss function
            print("Loading loss function...")
            self.criterion = nn.NLLLoss()
            
            # Load optimizer
            print("Loading the optimizer...")
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
            self.best_loss = sys.float_info.max
            
            # Load train & valid dataset
            print("Loading train & valid data...")
            train_set = CustomDataset('train', self.config)
            valid_set = CustomDataset('valid', self.config)
            self.train_loader = DataLoader(train_set, shuffle=True, batch_size=self.config['batch_size'])
            self.valid_loader = DataLoader(valid_set, shuffle=True, batch_size=self.config['batch_size'])
            
        if not os.path.exists(self.config['ckpt_dir']):
            os.mkdir(self.config['ckpt_dir'])
        
        if ckpt_name is not None:
            assert os.path.exists(f"{self.config['ckpt_dir']}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

            print("Loading checkpoint...")
            checkpoint = torch.load(f"{self.config['ckpt_dir']}/{ckpt_name}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if args.mode == 'train':
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
                self.best_loss = checkpoint['loss']
        else:
            print("Initializing the model...")
            self.model.init_model()
              
        print("Setting finished.")
              
    def train(self):
        print("Training starts.")
              
        for epoch in range(1, self.config['num_epochs']+1):
            self.model.train()
            
            print(f"#################### Epoch: {epoch} ####################")
            train_losses = []  
            for i, batch in enumerate(tqdm(self.train_loader)):
                src_inputs, trg_inputs, trg_outputs, e_mask, d_mask = batch
                src_inputs, trg_inputs, trg_outputs, e_mask, d_mask = \
                    src_inputs.to(self.config['device']), trg_inputs.to(self.config['device']), trg_outputs.to(self.config['device']), \
                    e_mask.to(self.config['device']), d_mask.to(self.config['device'])
              
                output = self.model(src_inputs, trg_inputs, e_mask, d_mask)  # (B, L, vocab_size)
                
                self.optim.zero_grad()
                
                loss = self.criterion(
                    output.view(-1, self.config['vocab_size']),
                    trg_outputs.contiguous().view(output.shape[0] * output.shape[1])
                )
                
                loss.backward()
                self.optim.step()
                
                train_losses.append(loss.item())
              
            mean_train_loss = np.mean(train_losses)
            print(f"Train loss: {mean_train_loss}")
            
            valid_loss = self.validation()
              
            if valid_loss < self.best_loss:
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'loss': self.best_loss,
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
                src_inputs, trg_inputs, trg_outputs, e_mask, d_mask = batch
                src_inputs, trg_inputs, trg_outputs, e_mask, d_mask = \
                    src_inputs.to(self.config['device']), trg_inputs.to(self.config['device']), trg_outputs.to(self.config['device']), \
                    e_mask.to(self.config['device']), d_mask.to(self.config['device'])
              
                output = self.model(src_inputs, trg_inputs, e_mask, d_mask)  # (B, L, vocab_size)
                
                loss = self.criterion(
                    output.view(-1, self.config['vocab_size']),
                    trg_outputs.contiguous().view(output.shape[0] * output.shape[1])
                )
                
                valid_losses.append(loss.item())
              
        mean_valid_loss = np.mean(valid_losses)
              
        return mean_valid_loss
        
              
    def inference(self):
        print("Let's start!")
        print(f"If you want to quit the conversation, please type \"{self.config['end_command']}\".")
        self.model.eval()
        
        with torch.no_grad():
            # Diagloue history
            init = [self.config['pad_id']] * self.config['max_len']
            history = [init for t in range(self.config['max_time'])]  # (T, L)
            
            utter = None
            output_ids = None
            for t in range(self.config['max_time']):
                if t % 2 == 0:
                    utter = input("You: ")
                    
                    if utter == self.config['end_command']:
                        print("Bot: Good bye.")
                        break

                    tokens = self.tokenizer.encode(utter)
                    if len(tokens) < self.config['max_len']:
                        sent = tokens + [self.config['eos_id']]
                        sent += [self.config['pad_id']] * (self.config['max_len'] - len(sent))
                    else:
                        sent = tokens[:self.config['max_len']]
                        sent[-1] = self.config['eos_id']

                    history[t] = sent
                    src_input = torch.LongTensor(history).unsqueeze(0).to(self.config['device'])  # (B, T, L)

                    src_emb = self.model.src_embed(src_input)  # (B, L, 2*d_model)
                    e_mask = [1 for i in range(t+1)] + [0 for i in range(self.config['max_time']-t-1)]
                    e_mask = torch.BoolTensor(e_mask).unsqueeze(0).unsqueeze(0).to(self.config['device'])  # (B, 1, T)
                    e_output = self.model.encoder(src_emb, e_mask)  # (B, L, 2*d_model)

                    output_ids = self.nucleus_sampling(e_output, e_mask)  # (L) list
                    res = self.tokenizer.decode(output_ids)

                    print(f"Bot: {res}")
                    
                else:
                    if len(output_ids) < self.config['max_len']:
                        sent = output_ids + [self.config['eos_id']]
                        sent += [self.config['pad_id']] * (self.config['max_len'] - len(sent))
                    else:
                        sent = output_ids[:self.config['max_len']]
                        sent[-1] = self.config['eos_id']
                        
                    history[t] = sent

                if t == self.config['max_time']-1:
                    print("This is the last turn.")

    def nucleus_sampling(self, e_output, e_mask):
        trg_input = [self.config['bos_id']]
        trg_input += [self.config['pad_id']] * (self.config['max_len']-len(trg_input))
        trg_input = torch.LongTensor(trg_input).unsqueeze(0).to(self.config['device'])  # (B, L)
        
        output_ids = []
        
        for pos in range(self.config['max_len']):
            trg_emb = self.model.trg_embed(trg_input)  # (B, L, 2*d_model)
            
            d_mask = (trg_input != self.config['pad_id']).unsqueeze(1)  # (B, 1, L)
            nopeak_mask = torch.ones([1, self.config['max_len'], self.config['max_len']], dtype=torch.bool)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask).to(self.config['device'])  # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask  # (B, L, L) padding false
            
            d_output = self.model.decoder(trg_emb, e_output, e_mask, d_mask)  # (B, L, 2*d_model)
            
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
            
        if output_ids[-1]== self.config['eos_id']:
            output_ids = output_ids[:-1]
            
        return output_ids
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help="Train or inference?")
    parser.add_argument('--use_gpt', required=False, help='Using GPT2 embedding layer?')
    parser.add_argument('--ckpt_name', required=False, help="Best checkpoint file.")
              
    args = parser.parse_args()
    
    assert args.mode == 'train' or args.mode=='inference', print("Please specify a correct mode name, 'train' or 'inference'.")
              
    if args.mode == 'train':
        if args.ckpt_name is not None:
            manager = Manager(args.mode, use_gpt=args.use_gpt, ckpt_name=args.ckpt_name)
        else:
            manager = Manager(args.mode, use_gpt=args.use_gpt)
              
        manager.train()
        
    elif args.mode == 'inference':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint."
        
        manager = Manager(args.mode, use_gpt=args.use_gpt, ckpt_name=args.ckpt_name)
        
        manager.inference()
