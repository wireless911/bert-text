from typing import Text, Dict, Optional

import torch


class TextClassifizerConfig(object):
    """Configuration for `TextClassifizer`."""

    def __init__(
            self,
            num_classes: int = 3,
            batch_size: int = 4,
            learning_rate: float = 1e-6,
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
    MAX_LENGTH = 250
    TAG_TO_ID = {
        PAD_TAG: 0,
        "B-person": 1,
        "I-person": 2,
        "B-mobile": 3,
        "I-mobile": 4,
        "B-provin": 5,
        "I-provin": 6,
        "B-cities": 7,
        "I-cities": 8,
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
            batch_size: int = 8,
            learning_rate: float = 5e-6,
            epochs: int = 50,
            max_length=MAX_LENGTH,
            hidden_dim=50,
            train_data: Text = "data/squence-label/train.csv",
            eval_data: Text = "data/squence-label/dev.csv",
            albert_vocab_file: Optional[Text] = "albert_base_zh/vocab_chinese.txt",
            albert_hidden_size: Optional[int] = 768,
            albert_pytorch_model_path: Optional[Text] = "models"

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
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.albert_vocab_file = albert_vocab_file
        self.albert_hidden_size = albert_hidden_size
        self.albert_pytorch_model_path = albert_pytorch_model_path
        self.tag_to_id = self.TAG_TO_ID