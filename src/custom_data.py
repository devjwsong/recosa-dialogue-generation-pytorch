from torch.utils.data import Dataset
from tqdm import tqdm

import torch


class CustomDataset(Dataset):
    def __init__(self, data_type, config):
        assert data_type == 'train' or data_type == 'valid', "Data type incorrect. It should be 'train' or 'valid'."
        
        self.config = config
        
        if data_type == 'train':
            data_name = self.config['train_name']
        elif data_type == 'valid':
            data_name = self.config['valid_name']
        
        print(f"Loading {data_name}_id.txt...")
        with open(f"{self.config['data_dir']}/{self.config['processed_dir']}/{data_name}_id.txt", 'r') as f:
            lines = f.readlines()
        
        self.dialogues = []  # (N, T, 3, L)
        
        print(f"Processing {data_name}_id.txt...")
        dialogue = []
        for i, line in enumerate(tqdm(lines)):
            if line.strip() == self.config['dialogue_split_line']:
                if len(dialogue) < self.config['max_turn']-1:
                    dummy = [self.config['pad_id']] * self.config['max_len']
                    dummies = [[dummy, dummy, dummy]] * (self.config['max_turn']-1-len(dialogue))
                    dialogue += dummies
                else:
                    dialogue = dialogue[:self.config['max_turn']-1]
                
                self.dialogues.append(dialogue)
                dialogue = []
            elif i+1<len(lines) and lines[i+1].strip() != self.config['dialogue_split_line']:
                src_sent = [int(token) for token in line.strip().split(' ')]
                trg_sent = [int(token) for token in lines[i+1].strip().split(' ')]
                
                src_input = self.process_src(src_sent)
                trg_input, trg_output = self.process_trg(trg_sent)
                    
                dialogue.append([src_input, trg_input, trg_output])
                
        self.dialogues = torch.LongTensor(self.dialogues)
            
    def process_src(self, src_sent):
        if len(src_sent) < self.config['max_len']:
            src_input = src_sent + [self.config['eos_id']]
            src_input += [self.config['pad_id']] * (self.config['max_len'] - len(src_input))
        else:
            src_input = src_sent[:self.config['max_len']]
            src_input[-1] = self.config['eos_id']
            
        return src_input
    
    def process_trg(self, trg_sent):
        if len(trg_sent) < self.config['max_len']:
            trg_output = trg_sent + [self.config['eos_id']]
            trg_output += [self.config['pad_id']] * (self.config['max_len'] - len(trg_output))
        else:
            trg_output = trg_sent[:self.config['max_len']]
            trg_output[-1] = self.config['eos_id']
            
        if len(trg_sent) < self.config['max_len']:
            trg_input = [self.config['bos_id']] + trg_sent
            trg_input += [self.config['pad_id']] * (self.config['max_len'] - len(trg_input))
        else:
            trg_input = [self.config['bos_id']] + trg_sent
            trg_input = trg_input[:self.config['max_len']]
            
        return trg_input, trg_output
    
    def __len__(self):
        return self.dialogues.shape[0]
    
    def __getitem__(self, idx):
        return self.dialogues[idx]
