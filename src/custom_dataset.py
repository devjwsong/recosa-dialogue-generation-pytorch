from torch.utils.data import Dataset
from tqdm import tqdm

import torch
import pickle
import json


class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, data_type):
        assert data_type in ["train", "valid", "test"]
        
        print(f"Loading {data_type} data...")
        with open(f"{args.task_dir}/{data_type}.pickle", "rb") as f:
            dials = pickle.load(f)
        
        with open(f"{args.task_dir}/data_info.json", "r") as f:
            data_info = json.load(f)
            
        self.src_idxs = []  # (N, T, S_L)
        self.num_valid_turns = []  # (N)
        self.trg_idxs = []  # (N, T_L)
        
        max_pers = data_info["max_num_pers"]
        num_contexts = max_pers + args.max_turns
        for dial in tqdm(dials):
            hists = []
            persona1, persona2, turns = dial['persona1'], dial['persona2'], dial['turns']
            
            pers = []  # The system's persona will be handled as extra histories without a speacker token. (or maybe empty...)
            for per in persona2:
                token_idxs = [args.bos_id] + tokenizer.encode(per) + [args.eos_id]
                pers.append(token_idxs)
            
            for t, turn in enumerate(turns):
                if t % 2 == 0:  # Speaker 1: User
                    token_idxs = [args.bos_id, args.sp1_id] + tokenizer.encode(turn) + [args.eos_id]
                else:  # Speacker 2: System
                    token_idxs = [args.bos_id, args.sp2_id] + tokenizer.encode(turn) + [args.eos_id]
                
                hists.append(token_idxs)
            
            hists = [self.trunc(token_idxs, args.src_max_len, args.eos_id) for token_idxs in hists]
            if len(pers) > 0:
                pers = [self.trunc(token_idxs, args.src_max_len, args.eos_id) for token_idxs in pers]
                
            for i in range(len(hists)):
                if i % 2 == 1:
                    self.trg_idxs.append(hists[i])
                    start, end = i-args.max_turns, i
                    if start < 0:
                        start = 0
                    context = hists[start:end]
                    assert len(context) > 0
                                        
                    if len(pers) > 0:
                        context = pers + context
                    
                    self.num_valid_turns.append(len(context))
                    
                    if len(context) < num_contexts:
                        num_extras = num_contexts - len(context)
                        context += [[args.bos_id, args.eos_id]] * num_extras
                    assert len(context) == num_contexts
                    
                    self.src_idxs.append(context)
        
        # Padding
        for c, context in enumerate(self.src_idxs):
            for i, utter in enumerate(self.src_idxs[c]):
                token_idxs = self.src_idxs[c][i]
                self.src_idxs[c][i] = self.padding(token_idxs, args.src_max_len, args.pad_id)
        
        assert len(self.src_idxs) == len(self.trg_idxs)
        assert len(self.src_idxs) == len(self.num_valid_turns)
    
    def __len__(self):
        return len(self.src_idxs)
    
    def __getitem__(self, idx):
        return self.src_idxs[idx], self.num_valid_turns[idx], self.trg_idxs[idx]
    
    def padding(self, token_idxs, max_len, pad_id):
        num_extras = max_len - len(token_idxs)
        token_idxs += [pad_id] * num_extras
        
        return token_idxs
    
    def trunc(self, token_idxs, max_len, eos_id):
        token_idxs = token_idxs[:max_len]
        token_idxs[-1] = eos_id
        
        return token_idxs
    
    
class PadCollate():
    def __init__(self, pad_id):
        self.pad_id = pad_id
        
    def pad_collate(self, batch):
        src_idxs, num_valid_turns, trg_idxs = [], [], []
        for seqs in batch:
            src_idxs.append(seqs[0])
            num_valid_turns.append(seqs[1])
            trg_idxs.append(torch.LongTensor(seqs[2]))

        trg_idxs = torch.nn.utils.rnn.pad_sequence(trg_idxs, batch_first=True, padding_value=self.pad_id)  # (B, T_L)
        
        try:
            return torch.LongTensor(src_idxs).contiguous(), torch.LongTensor(num_valid_turns).contiguous(), trg_idxs.contiguous()
        except:
            print(f"batch size: {len(src_idxs)}")
            for b in range(len(src_idxs)):
                print(f"num turns: {len(src_idxs[b])}")

            print(f"batch size: {len(num_valid_turns)}")
            print(num_valid_turns)

            print(trg_idxs.shape)
            exit()