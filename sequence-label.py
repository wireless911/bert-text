import collections
import time
import torch
from typing import Optional, Text
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train import SequenceLabelTrainer
from config import SequenceLabelConfig
from utils import CustomSequenceLabelDataset
from model import BiLSTM_CRF

config = SequenceLabelConfig()
tag_to_ix = SequenceLabelConfig.TAG_TO_ID

train_datasets = CustomSequenceLabelDataset(config.train_data, config)
eval_datasets = CustomSequenceLabelDataset(config.eval_data, config)

# create model
model = BiLSTM_CRF(tag_to_ix, config.max_sequence_length, config.hidden_dim)
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
    learning_rate=config.learning_rate
)

# train model
trainer.train()
