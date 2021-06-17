import time
import torch
from typing import Optional, Text
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from utils import CustomTextClassifizerDataset
from train import TextClassifizerTrainer
from model import TextClassificationModel
from config import TextClassifizerConfig

# load config from object
config = TextClassifizerConfig()

# load dataset
train_datasets = CustomTextClassifizerDataset(config.train_data)
eval_datasets = CustomTextClassifizerDataset(config.eval_data)

# dataloader
train_dataloader = DataLoader(train_datasets, batch_size=config.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_datasets, batch_size=config.batch_size, shuffle=True)

# create model
model = TextClassificationModel(config.max_sequence_length, config.num_classes)
# create trainer
trainer = TextClassifizerTrainer(
    model=model,
    args=None,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    epochs=config.epochs,
    learning_rate=config.learning_rate
)

# train model
trainer.train()
