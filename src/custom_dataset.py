from torch.utils.data import Dataset
from tqdm import tqdm

import torch
import pickle


class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, data_type):
        assert data_type in ["train", "valid", "test"]
        
        print(f"Loading {data_type} data...")
        with open(f"{args.task_dir}/{data_type}.pickle", "rb") as f:
            dials = pickle.load(f)
            
        self.src_idxs = []  # (N, T, S_L)
        self.num_valid_turns = []  # (N)
        self.trg_idxs = []  # (N, T_L)
        
        hists = []
        for dial in tqdm(dials):
            persona1, persona2, turns = dial['persona1'], dial['persona2'], dial['turns']
            
            pers = []  # The system's persona will be handled as extra histories without a speacker token. (or maybe empty...)
            for per in persona2:
                token_idxs = [args.bos_id] + tokenizer.encode(per) + [args.eos_id]
                pers.append(token_idxs)
            
            for t, turn in enumerate(turns):
                if t % 2 == 0:  # Speaker 1: User
                    token_idxs = [args.bos_id, args.sp1_id] + tokenizer.encode() + [args.eos_id]
                else:  # Speacker 2: System
                    token_idxs = [args.bos_id, args.sp2_id] + tokenizer.encode() + [args.eos_id]
                
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
                    
                    self.num_valid_turns.append(len(context))
                    
                    if len(context) < args.max_turns:
                        num_extras = args.max_turns - len(context)
                        context += [torch.LongTensor([args.bos_id, args.eos_id])] * num_extras
                    assert len(context) == args.max_turns
                    
                    if len(pers) > 0:
                        context = pers + context
                    self.src_idxs.append([self.padding(token_idxs, args.src_max_len, args.pad_id) for token_idxs in context])
        
        assert len(self.src_idxs) == len(self.trg_idxs)
        assert len(self.src_idxs) == len(self.num_valid_turns)
        
        self.src_idxs = torch.LongTensor(self.src_idxs)
    
    def __len__(self):
        return len(self.src_idxs[idx])
    
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
        src_idxs, num_valid_turns, trg_idxs = batch  # src_idxs: (B, T, S_L), num_valid_turns: (B), trg_idxs: (B, ?)
        trg_idxs = torch.nn.utils.rnn.pad_sequence(src_idxs, batch_first=True, padding_value=self.pad_id)  # (B, T_L)
        
        return src_idxs.contiguous(), torch.LongTensor(num_valid_turns).contiguous(), trg_idxs.contiguous()
        