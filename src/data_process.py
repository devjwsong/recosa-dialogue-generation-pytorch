from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from constants import *

import torch
import os
import numpy as np


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
        if 'Ġ' in token:
            if token[1:] in end_marks:
                token_list[i] = token[1:]

                if i < len(token_list)-1:
                    if token_list[i+1] not in end_marks and 'Ġ' not in token_list[i+1]:
                        token_list[i+1] = 'Ġ' + token_list[i+1]
            
            if token == 'Ġ'+quotes[1]:
                if i < len(token_list)-1:
                    if 'Ġ' in token_list[i+1] and token_list[i+1][1:] in abbreviations:
                        del_quote_idx.append(i)
                        token_list[i+1] = '\''+token_list[i+1][1:]
                    else:
                        token_list[i] = token_list[i][1:]
            
            if (double_quote_count % 2 == 0 and token == quotes[0]+'Ġ') \
                    or (double_quote_count % 2 == 1 and token == 'Ġ'+quotes[0]):
                token_list[i] = quotes[0]
                double_quote_count += 1
                
    if len(del_quote_idx) > 0:
        new_token_list = [token_list[i] for i, token in enumerate(token_list) if i not in del_quote_idx]
        token_list = new_token_list
        
    return token_list


def save_data(lines, tokenizer, name):
    total_string_list = []
    
    with open(f"{data_dir}/{processed_data_dir}/{name}.txt", 'w') as f: 
        for line in tqdm(lines):
            dialogue = line.strip().replace(' __eou__ ', '__eou__')
            dialogue = dialogue.replace(' __eou__', '__eou__')
            dialogue = dialogue.replace('__eou__ ', '__eou__')

            utters = dialogue.split('__eou__')[:-1]
            f.write(f"{dialogue_split_line}\n")
            
            total_string_list.append(f"{dialogue_split_line}")
            
            for utter in utters:
                token_list = tokenizer.tokenize(utter)
                token_list = process_token_list(token_list)
                
                total_string_list.append(' '.join(token_list))
                
                ids = []
                for token in token_list:
                    ids.append(str(tokenizer._convert_token_to_id(token)))
                    
                utter_idx = ' '.join(ids)
                f.write(f"{utter_idx}\n")
                
    with open(f"{data_dir}/{processed_data_dir}/string_{name}.txt", 'w') as f:
        for string in total_string_list:
            f.write(f"{string}\n")
        

class CustomDataset(Dataset):
    def __init__(self, dialogue_lines):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


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
    