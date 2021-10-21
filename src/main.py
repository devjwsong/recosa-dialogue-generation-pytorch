from transformers import *
from recosa_transformer import *
from custom_dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

import torch
import os, sys
import numpy as np
import argparse
import copy
import json


def train(args):
    # For directory setting
    args.task_dir = f"{args.data_dir}/{args.task_name}"
    assert os.path.isdir(args.task_dir)

    # Loading the pytorch lightning module
    print(f"Loading the pytorch lightning moduel for training...")
    module = TrainModule(args)
    
    # Loading datasets & dataloader
    train_set = CustomDataset(args, tokenizer, data_type="train")
    valid_set = CustomDataset(args, tokenizer, data_type="valid")
    test_set = CustomDataset(args, tokenizer, data_type="test")
    
    train_loader = DataLoader(train_set, collate_fn=ppd.pad_collate, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(valid_set, collate_fn=ppd.pad_collate, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(test_set, collate_fn=ppd.pad_collate, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)
    
    # Calculate total training steps
    args.gpus = [int(idx.strip()) for idx in args.gpus.split(",")]
    num_gpus = len(args.gpus)
    num_devices = num_gpus * args.num_nodes
    q, r = divmod(len(train_loader), num_devices)
    num_batches = q if r == 0 else q+1
    args.total_train_steps = args.num_epochs * num_batches
    args.warmup_steps = int(args.warmup_ratio * args.total_train_steps)
    
    print("Setting pytorch lightning callback & trainer...")
    # Model checkpoint & Early stopping callback
    filename = "{epoch}_{train_ppl:.4f}_{valid_ppl:.4f}"
    monitor = "valid_ppl"
    
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        verbose=True,
        monitor=monitor,
        mode='min',
        every_n_epochs=1,
        save_weights_only=True
    )
    
    stopping_callback = EarlyStopping(
        monitor=monitor,
        min_delta=1e-4,
        patience=3,
        verbose=True,
        mode='min'
    )
    
    # Trainer setting
    seed_everything(args.seed, workers=True)
    trainer = Trainer(
        check_val_every_n_epoch=1,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.max_grad_norm,
        deterministic=True,
        accelerator="ddp",
        callbacks=[checkpoint_callback, stopping_callback],
        plugins=DDPPlugin(find_unused_parameters=unused_param_flag),
    )
    
    print("Train starts.")
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print("Training done.")
    
    print("Test starts.")
    trainer.test(dataloaders=test_loader, ckpt_path='best')
    
    
def infer(args):
    print(f"Loading the pytorch lightning moduel for inference...")
    module = TrainModule.load_from_checkpoint(f"./lightning_logs/version_{args.log_idx}/checkpoints/{args.ckpt_file}")
    module.model.eval()
    args = module.args
    
    assert len(args.gpus.split(",")) == 1, "Inference should use only one GPU."
    device = torch.device(f"cuda:{self.args.gpus}")
    module = module.to(device)
    
    print("Let's start!")
    print(f"If you want to quit the conversation, please type \"{self.config['end_command']}\".")
    with torch.no_grad():
        hists = []
        utter, output_ids = None, None
        while True:
            utter = input("You: ")

            if utter == args.end_command:
                break

            token_idxs = [args.bos_id, args.sp1_id] + tokenizer.encode(utter) + [args.eos_id]
            if len(token_idxs) <= self.src_max_len:
                token_idxs += [args.pad_id] * (args.src_max_len - len(token_idxs))
            else:
                token_idxs = token_idxs[:self.src_max_len]
                token_idxs[-1] = self.eos_id

            assert len(token_idxs) == args.src_max_len

            hists.append(token_idxs)
            if len(hists) > args.max_turns:
                hists = hists[1:]

            num_valid_turns = torch.LongTensor([len(hists)])  # (1)
            src_idxs = torch.LongTensor(hists).unsqueeze(0).to(device)  # (1, T, S_L)

            output_ids = module(src_idxs, num_valid_turns)
            res = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            print(f"Bot: {res}")

            hists.append(output_ids)                
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help="The running mode: train or inference?")
    parser.add_argument('--seed', type=int, default=0, help="The random seed number.")
    parser.add_argument('--data_dir', type=str, default="data", help="The name of the parent directory where the whole data files are stored.")
    parser.add_argument('--task', type=str, required=True, help="The name of the specific task(dataset) name.")
    parser.add_argument('--pad_token', type=str, default="<pad>", help="The pad token.")
    parser.add_argument('--bos_token', type=str, default="<bos>", help="The bos token.")
    parser.add_argument('--eos_token', type=str, default="<eos>", help="The eos token.")
    parser.add_argument('--sp1_token', type=str, default="<sp1>", help="The speaker1 token.")
    parser.add_argument('--sp2_token', type=str, default="<sp2>", help="The speaker2 token.")
    parser.add_argument('--learning_rate', type=float, default=5e-4, help="The initial learning rate.")
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help="The warmup step ratio.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The max gradient for gradient clipping.")
    parser.add_argument('--train_batch_size', type=int, default=32, help="The batch size for training.")
    parser.add_argument('--eval_batch_size', type=int, default=8, help="The batch size for evaluating.")
    parser.add_argument('--num_epochs', type=int, default=10, help="The number of training epochs.")
    parser.add_argument('--src_max_len', type=int, default=128, help="The max length of each input utterance.")
    parser.add_argument('--src_max_turns', type=int, default=5, help="The max number of utterances to be included.")
    parser.add_argument('--trg_max_len', type=int, default=128, help="The max length of a target response.")
    parser.add_argument('--num_heads', type=int, default=8, help="The number of heads for multi-head attention.")
    parser.add_argument('--num_encoder_layers', type=int, default=6, help="The number of layers in the utterance-level encoder.")
    parser.add_argument('--num_gru_layers', type=int, default=2, help="The number of layers in the word-level encoder.")
    parser.add_argument('--gru_dropout', type=float, default=0.1, help="The dropout rate of the word-level encoder.")
    parser.add_argument('--num_decoder_layers', type=int, default=6, help="The number of layers in the decoder.")
    parser.add_argument('--d_model', type=int, default=768, help="The hidden size inside of the transformer module.")
    parser.add_argument('--d_pos', type=int, default=256, help="The hidden size of the positional embedding.")
    parser.add_argument('--d_ff', type=int, default=2048, help="The intermediate hidden size of each feed-forward layer.")
    parser.add_argument('--dropout', type=float, default=0.1, help="The dropout rate of the transformer modules.")
    parser.add_argument('--top_p', type=float, default=0.9, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument('--end_command', type=str, default="Abort!", help="The command to stop the conversation when inferencing.")
    parser.add_argument('--gpus', type=str, default="0", help="The indices of GPUs to use.")
    parser.add_argument('--num_nodes', type=int, default=1, help="The number of machine.")
    parser.add_argument('--log_idx', type=int, required=False, help="The index of a lightning log directory which contains the checkpoints to use.")
    parser.add_argument('--ckpt_file', type=str, required=False, help="The full name of the trained checkpoint for inferencing.")
    
    args = parser.parse_args()
    
    assert args.mode in ["train", "infer"]
    assert args.task in ["daily_dialog", "empathetic_dialogues", "persona_chat", "blended_skill_talk"]
    if args.mode == "infer":
        assert args.log_idx is not None
        
    print("#"*50 + "Running spec" + "#"*50)
    print(args)
              
    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args)
        
    print("GOOD BYE!")
