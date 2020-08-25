from torch.utils.data import Dataset

import torch


class CustomDataset(Dataset):
    def __init__(self, data_type, max_turn, max_len, pad_id, bos_id, eos_id, dialogue_split_line):
        assert data_type == 'train' or data_type == 'valid', "Data type incorrect. It should be 'train' or 'valid'."
        
        if data_type == 'train':
            data_name = train_name
        elif data_type == 'valid':
            data_name = valid_name
        
        print(f"Loading {data_name}_id.txt...")
        with open(f"{data_dir}/{processed_data_dir}/{data_name}_id.txt", 'r') as f:
            lines = f.readlines()
        
        self.max_turn = max_turn
        self.max_len = max_len,
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dialogue_split_line = dialogue_split_line
        self.dialogues = []  # (N, T, 3, L)
        
        print(f"Processing {data_name}_id.txt...")
        dialogue = []
        for i, line in tqdm(enumerate(lines)):
            if i+1<len(lines) and lines[i+1].strip() == dialogue_split_line:
                src_sent = [int(token) for token in line.strip().split(' ')]
                trg_sent = [int(token) for token in lines[i+1].strip().split(' ')]
                
                src_input = self.process_src(src_sent)
                trg_input, trg_output = self.process_trg(trg_sent)
                    
                dialogue.append([src_input, trg_input, trg_output])
                
            if line.strip() == dialogue_split_line:
                self.dialogues.append(dialogue)
                dialogue = []
                
        self.dialogues = torch.LongTensor(self.dialogues)
            
    def process_src(self, src_sent):
        if len(src_sent) < max_len:
            src_input = src_sent + [self.eos_id]
            src_input += [self.pad_id] * (max_len - len(src_input))
        else:
            src_input = src_sent[:self.max_len]
            src_input[-1] = self.eos_id
            
        return src_input
    
    def process_trg(self, trg_sent):
        if len(trg_sent) < max_len:
            trg_output = trg_sent + [self.eos_id]
            trg_output += [self.pad_id] * (max_len - len(trg_output))
        else:
            trg_output = trg_sent[:self.max_len]
            trg_output[-1] = self.eos_id
            
        if len(trg_sent) < max_len:
            trg_input = [self.bos_id] + trg_sent
            trg_input += [self.pad_id] * (max_len - len(trg_input))
        else:
            trg_input = [self.bos_id] + trg_sent
            trg_input = trg_input[:max_len]
            
        return trg_input, trg_output
    
    def __len__(self):
        return self.turn_nums.shape[0]
    
    def __getitem__(self, idx):
        return self.turn_nums[idx], self.dialogues[idx]
