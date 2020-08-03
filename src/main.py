from transformers import *
from tqdm import tqdm

import torch


# Parameters for data
data_dir = 'data'
raw_data_dir = 'raw'
processed_data_dir = 'processed'
train_name = 'train'
valid_name = 'validation'
test_name = 'test'
raw_name_prefix = 'dialogues'
train_frac = 0.8
end_of_utterance = '__eou__'
end_marks = ['.', ',', '?', '!']
quotes = ['"', '\'']
abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']
dialogue_split_line = "#################################"

