from typing import Text, Dict

import torch


class TextClassifizerConfig(object):
    """Configuration for `TextClassifizer`."""

    def __init__(
            self,
            num_classes: int = 3,
            batch_size: int = 256,
            learning_rate: float = 2e-5,
            epochs: int = 20,
            max_sequence_length: int = 100,
            train_data: Text = "data/text-classifizer/train.csv",
            eval_data: Text = "data/text-classifizer/dev.csv"
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.num_classes = num_classes
        self.train_data = train_data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_sequence_length = max_sequence_length


class SequenceLabelConfig(object):
    """Configuration for `SequenceLabelConfig`."""
    PAD_TAG = "[PAD]"
    MAX_LENGTH = 300
    TAG_TO_ID = {
        PAD_TAG: 0,
        "B-person": 1,
        "I-person": 2,
        "B-mobile": 3,
        "I-mobile": 4,
        "B-province": 5,
        "I-province": 6,
        "B-city": 7,
        "I-city": 8,
        "B-county": 9,
        "I-county": 10,
        "B-street": 11,
        "I-street": 12,
        "B-detail": 13,
        "I-detail": 14,
        "O": 15,
    }

    def __init__(
            self,
            batch_size: int = 32,
            learning_rate: float = 1e-5,
            epochs: int = 50,
            max_sequence_length=MAX_LENGTH,
            hidden_dim=50,
            train_data: Text = "data/squence-label/train.csv",
            eval_data: Text = "data/squence-label/dev.csv"

    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.train_data = train_data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = hidden_dim
