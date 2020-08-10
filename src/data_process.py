from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

import torch
import os
import numpy as np


# Parameters for data
data_dir = 'data'
raw_data_dir = 'raw'
processed_data_dir = 'processed'
train_name = 'train'
valid_name = 'validation'
test_name = 'test'
raw_name_prefix = 'dialogues'
train_frac = 0.8
pad = 'Ġ'
end_of_utterance = '__eou__'
end_marks = ['.', ',', '?', '!']
quotes = ['"', '\'']
abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']
dialogue_split_line = "#################################"


def merge_data(total_lines, data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        
    total_lines += lines
    
    return total_lines


def resplit_data(total_lines):
    train_lines = total_lines[:int(len(total_lines) * train_frac)]
    valid_lines = total_lines[int(len(total_lines) * train_frac):]
    
    return train_lines, valid_lines


def process_token_list(token_list):
    double_quote_count = 0
    del_quote_idx = []
    for i, token in enumerate(token_list):
        if pad in token:
            if token[1:] in end_marks:
                token_list[i] = token[1:]

                if i < len(token_list)-1:
                    if token_list[i+1] not in end_marks and pad not in token_list[i+1]:
                        token_list[i+1] = pad + token_list[i+1]
            
            if token == pad+quotes[1]:
                if i < len(token_list)-1:
                    if pad in token_list[i+1] and token_list[i+1][1:] in abbreviations:
                        del_quote_idx.append(i)
                        token_list[i+1] = '\''+token_list[i+1][1:]
                    else:
                        token_list[i] = token_list[i][1:]
            
            if (double_quote_count % 2 == 0 and token == quotes[0]+pad) \
                    or (double_quote_count % 2 == 1 and token == pad+quotes[0]):
                token_list[i] = quotes[0]
                double_quote_count += 1
                
    if len(del_quote_idx) > 0:
        new_token_list = [token_list[i] for i, token in enumerate(token_list) if i not in del_quote_idx]
        token_list = new_token_list
        
    return token_list


def save_data(lines, tokenizer, name):
    with open(f"{data_dir}/{processed_data_dir}/{name}.txt", 'w') as f: 
        for line in tqdm(lines):
            dialogue = line.strip().replace(' __eou__ ', '__eou__')
            dialogue = dialogue.replace(' __eou__', '__eou__')
            dialogue = dialogue.replace('__eou__ ', '__eou__')

            utters = dialogue.split('__eou__')[:-1]
            
            for utter in utters:
                token_list = tokenizer.tokenize(utter)
                token_list = process_token_list(token_list)
                
                ids = [str(tokenizer._convert_token_to_id(token)) for token in token_list]
                    
                utter_idx = ' '.join(ids)
                f.write(f"{utter_idx}\n")
                
            f.write(f"{dialogue_split_line}\n")


class CustomDataset(Dataset):
    def __init__(self, data_type, max_turn, max_len, pad_id):
        assert data_type == 'train' or data_type == 'valid', "Data type incorrect. It should be 'train' or 'valid'."
        
        if data_type == 'train':
            data_name = train_name
        elif data_type == 'valid':
            data_name = valid_name
        
        print(f"Loading {data_name}.txt...")
        with open(f"{data_dir}/{processed_data_dir}/{data_name}.txt", 'r') as f:
            lines = f.readlines()
        
        self.turn_nums = []  # (N)
        self.dialogues = []  # (N, T, L)
        
        print(f"Processing {data_name}.txt...")
        turn_num = 0
        dialogue = []
        for i, line in tqdm(enumerate(lines)):
            if line.strip() == dialogue_split_line:
                self.turn_nums.append(turn_num)
                turn_num = 0
                
                if len(dialogue) < max_turn:
                    left = max_turn - len(dialogue)
                    dummy = [pad_id] * max_len
                    for t in range(left):
                        dialogue.append(dummy)
            else:
                tokens = line.strip().split(' ')
                left = max_len - len(tokens)
                tokens += ([pad_id] * left)  # (L)
                dialogue.append(tokens)
                
        self.turn_nums = torch.LongTensor(self.turn_nums)
        self.dialogues = torch.LongTensor(self.dialogues)
            
    
    def __len__(self):
        return self.turn_nums.shape[0]
    
    def __getitem__(self, idx):
        return self.turn_nums[idx], self.dialogues[idx]


if __name__=='__main__':
    print("Merging all dialogue dataset...")
    total_lines = merge_data([], f"{data_dir}/{raw_data_dir}/dialogues_{train_name}.txt")
    total_lines = merge_data(total_lines, f"{data_dir}/{raw_data_dir}/dialogues_{valid_name}.txt")
    total_lines = merge_data(total_lines, f"{data_dir}/{raw_data_dir}/dialogues_{test_name}.txt")
    
    print("Respliting data...")
    train_lines, valid_lines = resplit_data(total_lines)
    
    print("Loading GPT2 Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    if not os.path.isdir(f"{data_dir}/{processed_data_dir}"):
        os.mkdir(f"{data_dir}/{processed_data_dir}")
    
    print("Processing train utterances...")
    save_data(train_lines, tokenizer, train_name)
    print("Processing valid utterances...")
    save_data(valid_lines, tokenizer, valid_name)            
    
    print("Data preprocess finished!")
