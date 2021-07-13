import torch
from config import TextClassifizerConfig, SequenceLabelConfig
from model import TextClassificationModel, BiLSTM_CRF
import torch.nn.functional as F
import pandas as pd

# load config from object
# config = TextClassifizerConfig()
#
# model = TextClassificationModel(config.max_sequence_length, 3)
#
# model.load_state_dict(torch.load('models/model-B128-E20-L2e-05.pkl'))
#
# model.eval()
# with torch.no_grad():
#     dataframe = pd.read_csv("data/text-classifizer/test.csv")
#     text_list = dataframe["text"]
#     label = dataframe["label"]
#     for idx,text in enumerate(text_list):
#         pred = model(text)
#         res = pred.argmax(1).item()
#         scores = F.softmax(pred)
#         print(f'lable{label[idx]} {res},scores:{scores.squeeze(0)[res]}')


config = SequenceLabelConfig()
tag_to_ix = SequenceLabelConfig.TAG_TO_ID
model = BiLSTM_CRF(tag_to_ix, config.max_sequence_length, config.hidden_dim)
model.summuary()
#
model.load_state_dict(torch.load('models/sequence-label-checkpoint_35_epoch.pkl')["model_state_dict"])
#
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

model.eval()
with torch.no_grad():
    dataframe = pd.read_csv("data/squence-label/dev.csv")
    text_list = dataframe["text"]
    label = dataframe["label"]
    for idx, text in enumerate(text_list):
        pred, padding_count = model(text)
        res = [ix_to_tag[x] for x in pred.squeeze(0).tolist()[:config.max_sequence_length - padding_count]]
        text = text.split(" ")
        print([f"{a}/{b}" for a, b in zip(text, res)])

