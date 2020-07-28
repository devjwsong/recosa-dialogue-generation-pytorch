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
dialogue_split_line = "#################################"

# Parameters for training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seq_len = 50
batch_size = 64
learning_rate = 0.0001
num_epochs = 10
nucleus_p = 0.95
ckpt_dir = 'saved_models'
