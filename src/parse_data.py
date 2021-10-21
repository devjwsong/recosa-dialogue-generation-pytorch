from tqdm import tqdm
from glob import glob

import pickle
import argparse
import os
import json


# For all
data_list = ['daily_dialog', 'empathetic_dialogues', 'persona_chat', 'blended_skill_talk']

# One dialogue consists of two persona lists + turns (Personas might be empty)
# {
#   persona1: [...],
#   persona2: [...],
#   turns: [...]
# }


def parse_daily_dialog(args):
    raw_dir = "ParlAI/data/dailydialog"
    save_dir = f"{args.data_dir}/daily_dialog"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    files = glob(f"{raw_dir}/*.json")
    
    num_train_dials, num_valid_dials, num_test_dials = 0, 0, 0
    num_train_utters, num_valid_utters, num_test_utters = 0, 0, 0
    for file in files:
        print(file)
        prefix = file.split("/")[-1].split(".")[0]  # "train" or "valid" or "test"
        assert prefix in ["train", "valid", "test"]
        
        dials = [json.loads(line) for line in open(file, 'r')]
        parsed_dials = []
        for dial in tqdm(dials):
            dialogue = dial['dialogue']
            turns = []
            for turn in dialogue:
                text = turn['text']
                turns.append(text)
            parsed_dials.append({"persona1": [], "persona2": [], "turns": turns})
        
        num_dials = len(parsed_dials)
        num_utters = 0
        for dial in parsed_dials:
            num_utters += len(dial["turns"])
            
        if prefix == "train":
            num_train_dials = num_dials
            num_train_utters = num_utters
        elif prefix == "valid":
            num_valid_dials = num_dials
            num_valid_utters = num_utters
        elif prefix == "test":
            num_test_dials = num_dials
            num_test_utters = num_utters
        
        with open(f"{save_dir}/{prefix}.pickle", 'wb') as f:
            pickle.dump(parsed_dials, f)
    
    return num_train_dials, num_valid_dials, num_test_dials, num_train_utters, num_valid_utters, num_test_utters
    
    
def parse_empathetic_dialogues(args):
    comma_symbol = "_comma_"
    
    raw_dir = "ParlAI/data/empatheticdialogues/empatheticdialogues"
    save_dir = f"{args.data_dir}/empathetic_dialogues"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    files = glob(f"{raw_dir}/*.csv")
    
    num_train_dials, num_valid_dials, num_test_dials = 0, 0, 0
    num_train_utters, num_valid_utters, num_test_utters = 0, 0, 0
    for file in files:
        print(file)
        prefix = file.split("/")[-1].split(".")[0]  # "train" or "valid" or "test"
        assert prefix in ["train", "valid", "test"]
        
        with open(file, 'r') as f:
            lines = f.readlines()
            
        cur_conv_id = ""
        dials, turns = [], []
        for l, line in enumerate(tqdm(lines)):
            comps = line.strip().split(',')
            
            if l == 0:
                continue
                
            if cur_conv_id != comps[0]:
                if len(turns) > 0:
                    dials.append({"persona1": [], "persona2": [], "turns": turns})
                turns = []
            else:
                assert int(comps[4]) != int(lines[l-1].strip().split(',')[4])
                
            cur_conv_id = comps[0]
            utter = comps[5]

            turns.append(utter.replace(comma_symbol, ","))
        
        if len(turns) > 0:
            dials.append({"persona1": [], "persona2": [], "turns": turns})
            
        num_dials = len(dials)
        num_utters = 0
        for dial in dials:
            num_utters += len(dial["turns"])
            
        if prefix == "train":
            num_train_dials = num_dials
            num_train_utters = num_utters
        elif prefix == "valid":
            num_valid_dials = num_dials
            num_valid_utters = num_utters
        elif prefix == "test":
            num_test_dials = num_dials
            num_test_utters = num_utters
        
        with open(f"{save_dir}/{prefix}.pickle", 'wb') as f:
            pickle.dump(dials, f)
    
    return num_train_dials, num_valid_dials, num_test_dials, num_train_utters, num_valid_utters, num_test_utters
    

def parse_persona_chat(args):
    raw_dir = "ParlAI/data/Persona-Chat/personachat"
    save_dir = f"{args.data_dir}/persona_chat"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    files = glob(f"{raw_dir}/*_self_original.txt")
    
    num_train_dials, num_valid_dials, num_test_dials = 0, 0, 0
    num_train_utters, num_valid_utters, num_test_utters = 0, 0, 0
    for file in files:
        print(file)
        prefix = file.split("/")[-1].split("_")[0]  # "train" or "valid" or "test"
        assert prefix in ["train", "valid", "test"]
        
        with open(file, 'r') as f:
            lines = f.readlines()
            
        cur_idx = 0
        dials, turns, pers = [], [], []
        for line in tqdm(lines):
            idx = line.strip().split(" ")[0]

            if cur_idx+1 != int(idx):
                assert int(idx) == 1
                if len(turns) > 0 and len(pers) > 0:
                    dials.append({"persona1": [], "persona2": pers, "turns": turns})
                turns, pers = [], []

            if "\t" in line: # utter
                sliced = line.strip()[len(idx):].strip()
                utters = sliced.split("\t\t")[0]
                turns += utters.split("\t")
            else: # persona
                persona = line.split("your persona:")[-1].strip()
                pers.append(persona)
            cur_idx = int(idx)
            
        if len(turns) > 0 and len(pers) > 0:
            dials.append({"persona1": [], "persona2": pers, "turns": turns})
            
        num_dials = len(dials)
        num_utters = 0
        for dial in dials:
            num_utters += len(dial["turns"])
            
        if prefix == "train":
            num_train_dials = num_dials
            num_train_utters = num_utters
        elif prefix == "valid":
            num_valid_dials = num_dials
            num_valid_utters = num_utters
        elif prefix == "test":
            num_test_dials = num_dials
            num_test_utters = num_utters
        
        with open(f"{save_dir}/{prefix}.pickle", 'wb') as f:
            pickle.dump(dials, f)
    
    return num_train_dials, num_valid_dials, num_test_dials, num_train_utters, num_valid_utters, num_test_utters


def parse_blended_skill_talk(args):
    raw_dir = "ParlAI/data/blended_skill_talk"
    save_dir = f"{args.data_dir}/blended_skill_talk"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    files = glob(f"{raw_dir}/*.json")
    files = [file for file in files if "_" not in file.split("/")[-1]]
    
    num_train_dials, num_valid_dials, num_test_dials = 0, 0, 0
    num_train_utters, num_valid_utters, num_test_utters = 0, 0, 0
    for file in files:
        print(file)
        prefix = file.split("/")[-1].split(".")[0]  # "train" or "valid" or "test"
        assert prefix in ["train", "valid", "test"]
        
        with open(file, 'r') as f:
            data = json.load(f)
        
        dials = []
        for dialogue in tqdm(data):
            personas = dialogue["personas"]
            persona1, persona2 = personas[0], personas[1]
            
            dial = {"persona1": persona1, "persona2": persona2}
            turns = [dialogue["free_turker_utterance"], dialogue["guided_turker_utterance"]]
            
            for turn in dialogue["dialog"]:
                turns.append(turn[-1])
                
            dial["turns"] = turns
            dials.append(dial)

        num_dials = len(dials)
        num_utters = 0
        for dial in dials:
            num_utters += len(dial["turns"])
            
        if prefix == "train":
            num_train_dials = num_dials
            num_train_utters = num_utters
        elif prefix == "valid":
            num_valid_dials = num_dials
            num_valid_utters = num_utters
        elif prefix == "test":
            num_test_dials = num_dials
            num_test_utters = num_utters
        
        with open(f"{save_dir}/{prefix}.pickle", 'wb') as f:
            pickle.dump(dials, f)
    
    return num_train_dials, num_valid_dials, num_test_dials, num_train_utters, num_valid_utters, num_test_utters

    

# def process_token_list(token_list):
#     space = 'Ġ'
#     pre_quote = '’'
#     end_marks = ['.', ',', '?', '!', '...']
#     quotes = ['"', '\'']
#     abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']
    
#     token_list[0] = token_list[0].capitalize()
    
#     quote_count = 0
#     for i, token in enumerate(token_list):
#         if space in token:
#             if token[1:] in end_marks or token[1:] in abbreviations:
#                 token_list[i] = token[1:]
                
#             if token[1:] == quotes[1]:
#                 if i<len(token_list)-1:
#                     if token_list[i+1] in abbreviations or (token_list[i+1][0] == space and token_list[i+1][1:] in abbreviations):
#                         token_list[i] = token[1:]
                        
#         if token[0] == space and token[1:] in quotes:
#             if quote_count % 2 == 1:
#                 token_list[i] = token[1:]
#                 quote_count = 0
#             else:
#                 if i<len(token_list)-1 and token_list[i+1][0] == space:
#                     token_list[i+1] = token_list[i+1][1:]
#                 quote_count += 1
                
#         if token in end_marks or token[1:] in end_marks:
#             if i<len(token_list)-1:
#                 if token_list[i+1][0] != space:
#                     token_list[i+1] = space + token_list[i+1].capitalize()
#                 else:
#                     token_list[i+1] = space + token_list[i+1][1:].capitalize()
                
#     new_token_list = [token for token in token_list if token != space and len(token)>0]
#     if new_token_list[-1] not in end_marks:
#         new_token_list.append(end_marks[0])
        
#     return new_token_list


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data", help="The name of the parent directory where the whole data files are stored.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    
    print(f"Data to download: {data_list}")
    
    print("Parsing each data...")
    total_num_train_dials, total_num_valid_dials, total_num_test_dials = 0, 0, 0
    total_num_train_utters, total_num_valid_utters, total_num_test_utters = 0, 0, 0
    for d, data in enumerate(data_list):
        print("#"*50 + f" {d+1}: {data} " + "#"*50)
    
        if data == 'daily_dialog':
            num_train_dials, num_valid_dials, num_test_dials, num_train_utters, num_valid_utters, num_test_utters = parse_daily_dialog(args)
        elif data == 'empathetic_dialogues':
            num_train_dials, num_valid_dials, num_test_dials, num_train_utters, num_valid_utters, num_test_utters = parse_empathetic_dialogues(args)
        elif data == 'persona_chat':
            num_train_dials, num_valid_dials, num_test_dials, num_train_utters, num_valid_utters, num_test_utters = parse_persona_chat(args)
        elif data == 'blended_skill_talk':
            num_train_dials, num_valid_dials, num_test_dials, num_train_utters, num_valid_utters, num_test_utters = parse_blended_skill_talk(args)
        
        print(f"The number of train dialogues: {num_train_dials}")
        print(f"The number of validation dialogues: {num_valid_dials}")
        print(f"The number of test dialogues: {num_test_dials}")
        print(f"The number of train utterances: {num_train_utters}")    
        print(f"The number of validation utterances: {num_valid_utters}")
        print(f"The number of test utterances: {num_test_utters}")
        
        total_num_train_dials += num_train_dials
        total_num_valid_dials += num_valid_dials
        total_num_test_dials += num_test_dials
        total_num_train_utters += num_train_utters
        total_num_valid_utters += num_valid_utters
        total_num_test_utters += num_test_utters
        
    print("#"*50 + "Total data statistics" + "#"*50)
    print(f"The number of train dialogues: {total_num_train_dials}")
    print(f"The number of validation dialogues: {total_num_valid_dials}")
    print(f"The number of test dialogues: {total_num_test_dials}")
    print(f"The number of train utterances: {total_num_train_utters}")    
    print(f"The number of validation utterances: {total_num_valid_utters}")
    print(f"The number of test utterances: {total_num_test_utters}")
    