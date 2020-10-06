from tqdm import tqdm
from transformers import *
from datasets import *

import torch
import os


# Parameters for data
data_dir = 'data'
train_name = 'train'
valid_name = 'validation'
train_frac = 0.85
space = 'Ġ'
pad = '<pad>'
unk = '<unk>'
bos = '<bos>'
eos = '<eos>'
dataset_list = ['daily_dialog']
dialogue_split_line = "[END OF DIALOGUE]"

pre_quote = '’'
end_marks = ['.', ',', '?', '!', '...']
quotes = ['"', '\'']
abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']


def load_daily_dialog():
    dataset = load_dataset('daily_dialog')
    train_dialogues = dataset['train']['dialog']
    valid_dialogues = dataset['validation']['dialog']
    test_dialogues = dataset['test']['dialog']
    
    train_utter_num = 0
    valid_utter_num = 0
    
    total_dialogues = train_dialogues + valid_dialogues + test_dialogues
    
    for i, dialogue in enumerate(tqdm(total_dialogues)):
        new_dialogue = []
        for utter in dialogue:
            token_list = tokenizer.tokenize(utter.strip().replace(pre_quote, quotes[1]))
            token_list = trim_daily_dialog(token_list)
            text = tokenizer.convert_tokens_to_string(token_list)
            new_dialogue.append(text)
            
        total_dialogues[i] = new_dialogue
        
    train_dialogues = total_dialogues[:int(len(total_dialogues)*train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues)*train_frac):]
    
    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)
        
    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)
    
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num
    

def trim_daily_dialog(token_list):
    quote_count = 0
    for i, token in enumerate(token_list):
        if space in token:
            if token[1:] in end_marks or token[1:] in abbreviations:
                token_list[i] = token[1:]
                
            if token[1:] == quotes[1]:
                if i<len(token_list)-1:
                    if token_list[i+1] in abbreviations or (token_list[i+1][0] == space and token_list[i+1][1:] in abbreviations):
                        token_list[i] = token[1:]
                        
        if token[0] == space and token[1:] in quotes:
            if quote_count % 2 == 1:
                token_list[i] = token[1:]
                quote_count = 0
            else:
                if i<len(token_list)-1 and token_list[i+1][0] == space:
                    token_list[i+1] = token_list[i+1][1:]
                quote_count += 1
                
        if token in end_marks or token[1:] in end_marks:
            if i<len(token_list)-1:
                if token_list[i+1][0] != space:
                    token_list[i+1] = space + token_list[i+1]
                
    new_token_list = [token for token in token_list if token != space and len(token)>0]
        
    return new_token_list


def save_data(dialogues, name):
    texts = []
    ids = []
    for dialogue in tqdm(dialogues):
        dialogue_ids = []
        for utter in dialogue:   
            texts.append(utter)
            token_ids = tokenizer(utter)['input_ids']
            dialogue_ids.append(token_ids)
        
        texts.append(dialogue_split_line)
        ids.append(dialogue_ids)
    
    print(f"Saving {name} text file...")
    with open(f"{data_dir}/{name}.txt", 'w') as f:
        for text in tqdm(texts):
            f.write(f"{text}\n")
    
    print(f"Saving {name} id file...")
    with open(f"{data_dir}/{name}_id.txt", 'w') as f:
        for dialogue in tqdm(ids):
            for utter in dialogue:
                utter_str = [str(idx) for idx in utter]
                f.write(f"{' '.join(utter_str)}\n")
            f.write(f"{dialogue_split_line}\n")


if __name__=='__main__':
    print("Loading the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {
        'bos_token': bos,
        'eos_token': eos,
        'pad_token': pad,
        'unk_token': unk
    }
    tokenizer.add_special_tokens(special_tokens)
    
    print("Loading & Merging all datasets...")
    train_dialogues = []
    valid_dialogues = []
    for data_name in dataset_list:
        if data_name=='daily_dialog':
            partial_train_dialogues, partial_valid_dialogues, train_utter_num, valid_utter_num = load_daily_dialog()
        
        train_dialogues += partial_train_dialogues
        valid_dialogues += partial_valid_dialogues
    
        print(f"#################### Analysis on {data_name} ####################")
        print(f"The number of train dialogues: {len(train_dialogues)}")
        print(f"The number of valid dialogues: {len(valid_dialogues)}")    
        print(f"The number of train utterances: {train_utter_num}")    
        print(f"The number of valid utterances: {valid_utter_num}")    
    
    if not os.path.isdir(f"{data_dir}"):
        os.mkdir(f"{data_dir}")
    
    print("Saving train data...")
    save_data(train_dialogues, train_name)
    print("Saving validation data...")
    save_data(valid_dialogues, valid_name)            
    
    print("Data preprocess finished!")
