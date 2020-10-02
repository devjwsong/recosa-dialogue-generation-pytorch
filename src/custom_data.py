from torch.utils.data import Dataset
from tqdm import tqdm

import torch


class CustomDataset(Dataset):
    def __init__(self, data_type, config):
        assert data_type == 'train' or data_type == 'valid', "Data type incorrect. It should be 'train' or 'valid'."
        
        if data_type == 'train':
            data_name = config['train_name']
        elif data_type == 'valid':
            data_name = config['valid_name']
        
        print(f"Loading {data_name}_id.txt...")
        with open(f"{config['data_dir']}/{config['processed_dir']}/{data_name}_id.txt", 'r') as f:
            lines = f.readlines()[:1000]
        
        self.src_inputs = []  # (N, T, L)
        self.trg_inputs = []  # (N, L)
        self.trg_outputs = []  # (N, L)
        
        self.e_masks = []  # (N, 1, T)
        
        print(f"Processing {data_name}_id.txt...")
        init = [config['pad_id']] * config['max_len']
        history = [init for t in range(config['max_turn'])]  # (T, L)
        num_turn = 0
        for i, line in enumerate(tqdm(lines)):
            if line.strip() == config['dialogue_split_line']:
                history = [init for t in range(config['max_turn'])]
                num_turn = 0
            elif i+1<len(lines) and lines[i+1].strip() != config['dialogue_split_line']:                    
                if num_turn < config['max_turn']:
                    src_sent = [int(token) for token in line.strip().split(' ')]
                    trg_sent = [int(token) for token in lines[i+1].strip().split(' ')]

                    src_input = self.process_src(src_sent, config['max_len'], config['bos_id'], config['eos_id'], config['pad_id'])
                    trg_input, trg_output = self.process_trg(trg_sent, config['max_len'], config['bos_id'], config['eos_id'], config['pad_id'])
                    
                    history[num_turn] = src_input
                    e_mask = self.make_encoder_mask(num_turn, config['max_turn'])
                    num_turn += 1
                    
                    self.src_inputs.append(history)
                    self.trg_inputs.append(trg_input)
                    self.trg_outputs.append(trg_output)
                    
                    self.e_masks.append(e_mask)
                    
        self.src_inputs = torch.LongTensor(self.src_inputs)  # (N, T, L)
        self.trg_inputs = torch.LongTensor(self.trg_inputs)  # (N, L)
        self.trg_outputs = torch.LongTensor(self.trg_outputs)  # (N, L)
        
        self.e_masks = torch.BoolTensor(self.e_masks).unsqueeze(1)  # (N, 1, T)
        self.d_masks = self.make_decoder_mask(self.trg_inputs, config['pad_id'], config['max_len'])  # (N, L, L)
        
            
    def process_src(self, src_sent, max_len, bos_id, eos_id, pad_id):
        if len(src_sent) < max_len:
            src_input = src_sent + [eos_id]
            src_input += [pad_id] * (max_len - len(src_input))
        else:
            src_input = src_sent[:max_len]
            src_input[-1] =eos_id
            
        return src_input
    
    def process_trg(self, trg_sent, max_len, bos_id, eos_id, pad_id):
        if len(trg_sent) < max_len:
            trg_output = trg_sent + [eos_id]
            trg_output += [pad_id] * (max_len - len(trg_output))
        else:
            trg_output = trg_sent[:max_len]
            trg_output[-1] = eos_id
            
        if len(trg_sent) < max_len:
            trg_input = [bos_id] + trg_sent
            trg_input += [pad_id] * (max_len - len(trg_input))
        else:
            trg_input = [bos_id] + trg_sent
            trg_input = trg_input[:max_len]
            
        return trg_input, trg_output
    
    def __len__(self):
        return self.src_inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.src_inputs[idx], self.trg_inputs[idx], self.trg_outputs[idx], \
            self.e_masks[idx], self.d_masks[idx]
    
    def make_encoder_mask(self, num_turn, max_turn):
        e_mask = [1 for i in range(num_turn+1)] + [0 for i in range(max_turn-num_turn-1)]
        
        return e_mask
    
    def make_decoder_mask(self, trg_inputs, pad_id, max_len):
        d_masks = (trg_inputs != pad_id).unsqueeze(1)  # (N, 1, L)

        nopeak_mask = torch.ones([1, max_len, max_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_masks = d_masks & nopeak_mask  # (N, L, L) padding false
        
        return d_masks
