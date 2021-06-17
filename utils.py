import collections
import os
import re
from typing import Text, Optional, Dict, Set
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import SequenceLabelConfig


class CustomTextClassifizerDataset(Dataset):
    """classifizer dataset"""

    def __init__(self, filepath):
        self.dataframe = pd.read_csv(filepath)
        self.text_dir = filepath

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        labels = self.dataframe.iloc[idx, 0]
        text = self.dataframe.iloc[idx, 1]
        item = {"labels": labels, "text": text}
        return item

    @property
    def num_classes(self) -> int:
        return len(set(self.dataframe["label"]))


class CustomSequenceLabelDataset(Dataset):
    """sequence label dataset"""

    def __init__(self, filepath, config: SequenceLabelConfig):
        self.dataframe = pd.read_csv(filepath)
        self.text_dir = filepath
        self.max_length = config.max_sequence_length
        self.tag2idx = config.TAG_TO_ID
        self.pad_tag = config.PAD_TAG

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx, 0]
        labels = self.dataframe.iloc[idx, 1]
        labels = [self.tag2idx[t] for t in labels.split(" ")]
        padding_length = self.max_length - len(labels)
        padding_list = [self.tag2idx[self.pad_tag]] * padding_length
        pad = torch.tensor(padding_list)
        labels = torch.cat((torch.tensor(labels), pad), dim=-1)
        item = {"labels": labels, "text": text}
        return item
