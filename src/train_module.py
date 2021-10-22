from torch import nn as nn
from pytorch_lightning import seed_everything
from argparse import Namespace
from transformers import get_linear_schedule_with_warmup, GPT2Tokenizer
from recosa_transformer import *

import torch
import pytorch_lightning as pl
import time
import math
import numpy as np


class TrainModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
            
        self.args = args
        
        # Tokenzier
        print("Setting a GPT2 tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        special_tokens = {
            'bos_token': args.bos_token,
            'eos_token': args.eos_token,
            'pad_token': args.pad_token,
            'additional_special_tokens': [args.sp1_token, args.sp2_token]
        }
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        args.vocab_size = len(vocab)
        args.pad_id = vocab[args.pad_token]
        args.bos_id = vocab[args.bos_token]
        args.eos_id = vocab[args.eos_token]
        args.sp1_id = vocab[args.sp1_token]
        args.sp2_id = vocab[args.sp2_token]
        
        # Model
        print("Initializing the model...")
        seed_everything(args.seed, workers=True)
        self.model = ReCoSaTransformer(args)
        self.model.init_model()
        
        # Loss function
        self.loss_func = nn.CrossEntropyLoss(ignore_index=args.pad_id)
        
        self.save_hyperparameters(args)
        
    def forward(self, src_idxs, num_valid_turns):  # (1, T, S_L), (1)
        num_contexts = src_idxs.shape[1]
        src_poses = torch.arange(num_contexts, device=src_idxs.device).unsqueeze(0) # (1, T)
        src_embs = self.model.src_embedding(src_idxs, src_poses)  # (1, T, d_model)
        e_masks = self.make_encoder_mask(num_valid_turns, num_contexts)  # (1, 1, T)
        e_outputs = self.model.encoder(src_embs, e_masks)  # (1, T, d_model)
        
        output_ids = self.nucleus_sampling(e_outputs, e_masks)
        
        return output_ids
    
    def training_step(self, batch, batch_idx):
        src_idxs, num_valid_turns, trg_idxs = batch  # src_idxs: (B, T, S_L), num_valid_turns: (B), trg_idxs: (B, T_L)
        batch_size, num_contexts, trg_len = src_idxs.shape[0], src_idxs.shape[1], trg_idxs.shape[1]
        e_masks = self.make_encoder_mask(num_valid_turns, num_contexts)  # (B, 1, T)
        d_masks = self.make_decoder_mask(trg_idxs[:, :-1], self.args.pad_id)  # (B, T_L, T_L)
        src_poses = torch.arange(num_contexts, device=src_idxs.device).unsqueeze(0).expand(batch_size, num_contexts)  # (B, T)
        trg_poses = torch.arange(trg_len-1, device=trg_idxs.device).unsqueeze(0).expand(batch_size, trg_len-1)  # (B, T_L)
        
        outputs = self.model(src_idxs, trg_idxs[:, :-1], src_poses, trg_poses, e_masks, d_masks)  # (B, T_L, V)
        
        preds = torch.max(outputs, dim=-1).indices  # (B, T_L)
        loss = self.loss_func(outputs.contiguous().view(-1, self.args.vocab_size), trg_idxs[:, 1:].contiguous().view(-1))
        ppl = torch.exp(loss)
        
        return {'loss': loss, 'ppl': ppl.detach()}
    
    def training_epoch_end(self, training_step_outputs):
        train_ppls = []
        for result in training_step_outputs:
            if math.isnan(result['ppl']):
                train_ppls.append(torch.LongTensor([1e+8]))
            else:
                train_ppls.append(result['ppl'])
        
        train_ppls = [ppl.item() for ppl in train_ppls]
        train_ppl = np.mean(train_ppls)
        
        self.log('train_ppl', train_ppl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def validation_step(self, batch, batch_idx):
        src_idxs, num_valid_turns, trg_idxs = batch  # src_idxs: (B, T, S_L), num_valid_turns: (B), trg_idxs: (B, T_L)
        batch_size, num_contexts, trg_len = src_idxs.shape[0], src_idxs.shape[1], trg_idxs.shape[1]
        e_masks = self.make_encoder_mask(num_valid_turns, num_contexts)  # (B, 1, T)
        d_masks = self.make_decoder_mask(trg_idxs[:, :-1], self.args.pad_id)  # (B, T_L, T_L)
        src_poses = torch.arange(num_contexts, device=src_idxs.device).unsqueeze(0).expand(batch_size, num_contexts)  # (B, T)
        trg_poses = torch.arange(trg_len-1, device=trg_idxs.device).unsqueeze(0).expand(batch_size, trg_len-1)  # (B, T_L)
        
        outputs = self.model(src_idxs, trg_idxs[:, :-1], src_poses, trg_poses, e_masks, d_masks)  # (B, T_L, V)
        
        preds = torch.max(outputs, dim=-1).indices  # (B, T_L)
        loss = self.loss_func(outputs.contiguous().view(-1, self.args.vocab_size), trg_idxs[:, 1:].contiguous().view(-1))
        ppl = torch.exp(loss)
        
        return {'loss': loss.detach(), 'ppl': ppl.detach()}
    
    def validation_epoch_end(self, validation_step_outputs):
        valid_ppls = []
        for result in validation_step_outputs:
            if math.isnan(result['ppl']):
                valid_ppls.append(torch.LongTensor([1e+8]))
            else:
                valid_ppls.append(result['ppl'])
        
        valid_ppls = [ppl.item() for ppl in valid_ppls]
        valid_ppl = np.mean(valid_ppls)
        
        self.log('valid_ppl', valid_ppl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        src_idxs, num_valid_turns, trg_idxs = batch  # src_idxs: (B, T, S_L), num_valid_turns: (B), trg_idxs: (B, T_L)
        batch_size, num_contexts, trg_len = src_idxs.shape[0], src_idxs.shape[1], trg_idxs.shape[1]
        e_masks = self.make_encoder_mask(num_valid_turns, num_contexts)  # (B, 1, T)
        d_masks = self.make_decoder_mask(trg_idxs[:, :-1], self.args.pad_id)  # (B, T_L, T_L)
        src_poses = torch.arange(num_contexts, device=src_idxs.device).unsqueeze(0).expand(batch_size, num_contexts)  # (B, T)
        trg_poses = torch.arange(trg_len-1, device=trg_idxs.device).unsqueeze(0).expand(batch_size, trg_len-1)  # (B, T_L)
        
        outputs = self.model(src_idxs, trg_idxs[:, :-1], src_poses, trg_poses, e_masks, d_masks)  # (B, T_L, V)
        
        preds = torch.max(outputs, dim=-1).indices  # (B, T_L)
        loss = self.loss_func(outputs.contiguous().view(-1, self.args.vocab_size), trg_idxs[:, 1:].contiguous().view(-1))
        ppl = torch.exp(loss)
        
        return {'loss': loss.detach(), 'ppl': ppl.detach()}
    
    def test_epoch_end(self, test_step_outputs):
        test_ppls = [] 
        for result in test_step_outputs:
            if math.isnan(result['ppl']):
                test_ppls.append(torch.LongTensor([1e+8]))
            else:
                test_ppls.append(result['ppl'])
        
        test_ppls = [ppl.item() for ppl in test_ppls]
        test_ppl = np.mean(test_ppls)
        
        self.log('test_ppl', test_ppl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        if self.args.warmup_steps < 0.0:
            return [optimizer]
        else:
            scheduler = {
                'scheduler': get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.args.warmup_steps,
                    num_training_steps=self.args.total_train_steps
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1

            }

            return [optimizer], [scheduler]
        
    def make_encoder_mask(self, num_valid_turns, num_contexts):
        batch_size = num_valid_turns.shape[0]
        masks = torch.zeros((num_valid_turns.shape[0], num_contexts), device=num_valid_turns.device)
        masks[torch.arange(num_contexts, device=masks.device) < num_valid_turns[..., None]] = 1.0
        
        return masks.bool().unsqueeze(1)  # (B, 1, T)
    
    def make_decoder_mask(self, trg_idxs, pad_id):
        padding_masks = (trg_idxs != pad_id).unsqueeze(1)  # (B, 1, T_L)
        
        max_len = trg_idxs.shape[1]
        nopeak_masks = torch.ones([1, max_len, max_len], dtype=torch.bool, device=padding_masks.device)  # (1, T_L, T_L)
        nopeak_masks = torch.tril(nopeak_masks)  # (1, T_L, T_L)
        
        return padding_masks & nopeak_masks  # (B, T_L, T_L)
    
    def nucleus_sampling(self, e_outputs, e_masks):
        trg_input = torch.full((self.args.trg_max_len, ), self.args.pad_id, dtype=torch.long, device=e_outputs.device)  # (T_L)
        trg_input[0] = self.args.bos_id
        trg_input[1] = self.args.sp2_id

        trg_input = trg_input.unsqueeze(0)  # (1, T_L)
        trg_pos = torch.arange(self.args.trg_max_len, device=trg_input.device).unsqueeze(0) # (1, T_L)
        
        output_ids = []
        
        for pos in range(1, self.args.trg_max_len):
            trg_emb = self.model.trg_embedding(trg_input, trg_pos)  # (1, L, d_model)
            d_mask = self.make_decoder_mask(trg_input, self.args.pad_id)  # (1, T_L, T_L)
            
            d_output = self.model.decoder(trg_emb, e_outputs, e_masks, d_mask)  # (1, T_L, d_model)
            
            output = F.softmax(self.model.output_linear(d_output), dim=-1)  # (1, T_L, V)
            output = output[:, pos]  # (1, V)
            
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, V)
            idx_remove = cumsum_probs > self.args.top_p
            sorted_probs[idx_remove] = 1e-8
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (1, V)
            
            # Random sampling
            probs = torch.zeros(output.shape, device=output.device).scatter_(-1, sorted_idxs, sorted_probs)  # (1, V)
            idx = torch.multinomial(probs, 1).squeeze(-1)  # (1)
            
            if pos < self.args.trg_max_len-1:
                trg_input[:, pos+1] = idx
            
            output_ids.append(idx.squeeze(0).item())    
            if idx.squeeze(0).item() == self.args.eos_id:
                break
            
        if output_ids[-1] != self.args.eos_id:
            output_ids[-1] = self.args.eos_id
            
        return output_ids
        