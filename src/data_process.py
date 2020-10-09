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
dataset_list = ['daily_dialog', 'empathetic_dialogues']
dialogue_split_line = "[END OF DIALOGUE]"

# For daily dialog
pre_quote = '’'
end_marks = ['.', ',', '?', '!', '...']
quotes = ['"', '\'']
abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']

# For empathetic dialogues
exclude_symbol = "_conv"
comma_symbol = "_comma_"


#https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json

def load_daily_dialog():
    dataset = load_dataset('daily_dialog')
    train_dialogues = dataset['train']['dialog']
    valid_dialogues = dataset['validation']['dialog']
    test_dialogues = dataset['test']['dialog']
    
    total_dialogues = train_dialogues + valid_dialogues + test_dialogues
    
    for i, dialogue in enumerate(tqdm(total_dialogues)):
        new_dialogue = []
        for utter in dialogue:
            token_list = tokenizer.tokenize(utter.strip().replace(pre_quote, quotes[1]))
            token_list = process_daily_dialog(token_list)
            text = tokenizer.convert_tokens_to_string(token_list)
            new_dialogue.append(text)
            
        total_dialogues[i] = new_dialogue
    
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = total_dialogues[:int(len(total_dialogues)*train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues)*train_frac):]
    
    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)
        
    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)
    
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num
    
    
def load_empathetic_dialogues():
    dataset = load_dataset('empathetic_dialogues')
    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']
    
    total_utters = train_data['utterance'] + valid_data['utterance'] + test_data['utterance']
    total_conv_ids = train_data['conv_id'] + valid_data['conv_id'] + test_data['conv_id']
    total_speaker_ids = train_data['speaker_idx'] + valid_data['speaker_idx'] + test_data['speaker_idx']
    
    assert len(total_utters) == len(total_conv_ids) and len(total_conv_ids) == len(total_speaker_ids)
    
    num = 0
    
    conv_dict = {}
    cur_speaker_idx = -1
    for i, utter in enumerate(tqdm(total_utters)):
        conv_id = total_conv_ids[i]
        speaker_idx = total_speaker_ids[i]
        
        utter_modified = utter.replace(comma_symbol, ',')
        
        if exclude_symbol in utter:
            continue
        
        if conv_id not in conv_dict:
            conv_dict[conv_id] = []
            cur_speaker_idx = -1

        if cur_speaker_idx != speaker_idx:
            conv_dict[conv_id].append(utter_modified)
            cur_speaker_idx = speaker_idx
        else:
            conv_dict[conv_id][-1] += f" {utter_modified}"
    
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = []
    valid_dialogues = []
    
    train_dialogue_num = int(len(conv_dict) * train_frac)
    for i, (conv_id, utter_list) in enumerate(conv_dict.items()):
        if i < train_dialogue_num:
            train_utter_num += len(utter_list)
            train_dialogues.append(utter_list)
        else:
            valid_utter_num += len(utter_list)
            valid_dialogues.append(utter_list)
            
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num
    

def process_daily_dialog(token_list):
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
    config = GPT2Config()
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
    total_train_dialogue_num = 0
    total_valid_dialogue_num = 0
    total_train_utter_num = 0
    total_valid_utter_num = 0
    for data_name in dataset_list:
        print(f"Processing {data_name}...")
        if data_name == 'daily_dialog':
            partial_train_dialogues, partial_valid_dialogues, train_utter_num, valid_utter_num = load_daily_dialog()
        elif data_name == 'empathetic_dialogues':
            partial_train_dialogues, partial_valid_dialogues, train_utter_num, valid_utter_num = load_empathetic_dialogues()
        
        train_dialogues += partial_train_dialogues
        valid_dialogues += partial_valid_dialogues
    
        print(f"#################### Analysis on {data_name} ####################")
        print(f"The number of train dialogues: {len(partial_train_dialogues)}")
        print(f"The number of valid dialogues: {len(partial_valid_dialogues)}")    
        print(f"The number of train utterances: {train_utter_num}")    
        print(f"The number of valid utterances: {valid_utter_num}")
        
        total_train_dialogue_num += len(train_dialogues)
        total_valid_dialogue_num += len(valid_dialogues)
        total_train_utter_num += train_utter_num
        total_valid_utter_num += valid_utter_num
    
    if not os.path.isdir(f"{data_dir}"):
        os.mkdir(f"{data_dir}")
    
    print("Saving train data...")
    save_data(train_dialogues, train_name)
    print("Saving validation data...")
    save_data(valid_dialogues, valid_name)            
    
    print("Data preprocess finished!")

    print(f"#################### Analysis on total data ####################")
    print(f"The number of train dialogues: {total_train_dialogue_num}")
    print(f"The number of valid dialogues: {total_valid_dialogue_num}")    
    print(f"The number of train utterances: {total_train_utter_num}")    
    print(f"The number of valid utterances: {total_valid_utter_num}")
    