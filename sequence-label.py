import collections
import time
import torch
from typing import Optional, Text
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from trainer import SequenceLabelTrainer
from config import SequenceLabelConfig
from utils import CustomSequenceLabelDataset
from model import BiLSTM_CRF

config = SequenceLabelConfig()
tag_to_ix = SequenceLabelConfig.TAG_TO_ID
# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_datasets = CustomSequenceLabelDataset(config.train_data, tokenizer, config)
eval_datasets = CustomSequenceLabelDataset(config.eval_data, tokenizer, config)

# create model
model = BiLSTM_CRF(tag_to_ix, config.max_sequence_length, config.hidden_dim,config.device)
model.summuary()

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.1)

# dataloader
train_dataloader = DataLoader(train_datasets, batch_size=config.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_datasets, batch_size=config.batch_size, shuffle=True)

# create trainer
trainer = SequenceLabelTrainer(
    model=model,
    args=None,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    epochs=config.epochs,
    learning_rate=config.learning_rate,
    device=config.device,
    padding_tag=config.TAG_TO_ID[config.PAD_TAG]
)

# train model
trainer.train()
