from typing import Optional, Text

import torch
from transformers import BertTokenizer

from config import TextClassifizerConfig, SequenceLabelConfig
from model import TextClassificationModel, BiLSTM_CRF
import torch.nn.functional as F
import pandas as pd
import re
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

# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
config = SequenceLabelConfig()
tag_to_ix = SequenceLabelConfig.TAG_TO_ID
model = BiLSTM_CRF(tag_to_ix, config.max_length, config.hidden_dim,config.device)
model.summuary()
model.to(config.device)
#
model.load_state_dict(torch.load('models/sequence-label-checkpoint_7_epoch.pkl')["model_state_dict"])
#
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

model.eval()
# with torch.no_grad():
#     dataframe = pd.read_csv("data/squence-label/dev.csv")
#     text_list = dataframe["text"]
#     label = dataframe["label"]
#     for idx, text in enumerate(text_list):
#         pred, padding_count = model(text)
#         res = [ix_to_tag[x] for x in pred.squeeze(0).tolist()[:config.max_length - padding_count]]
#         text = text.split(" ")
#         print([f"{a}/{b}" for a, b in zip(text, res)])

def predict(text:Optional[Text]):
    with torch.no_grad():
        curr_text = text
        curr_text = curr_text.replace(" ","")
        text_list = [x for x in curr_text]
        text = " ".join(text_list)
        token = tokenizer(text, return_tensors='pt', padding="max_length", max_length=config.max_length,
                               truncation=True)
        input_ids = token["input_ids"].squeeze(1).to(config.device)
        attention_mask = token["attention_mask"].squeeze(1).to(config.device)
        token_type_ids = token["token_type_ids"].squeeze(1).to(config.device)

        # Compute prediction and loss
        y_pred = model(input_ids, attention_mask, token_type_ids)
        res = [ix_to_tag[x] for x in y_pred.squeeze(0).tolist()]
        # print([f"{a}/{b}" for a, b in zip(text_list, res)])
        label = "".join(res)
        label = label.replace("O", "U-meamea")
        pattern = re.compile(
            "B-person*(I-person)*|B-mobile*(I-mobile)|B-provin*(I-provin)*|B-cities*(I-cities)*|B-county*(I-county)*|B-street*(I-street)*|B-detail*(I-detail)*")
        resutlt = re.finditer(pattern, label)
        result_word = {"province": [], "city": [], "county": [], "street": [],
                       "detail": [], "person": [], "cellphones": []}
        shiti_dict = {"province": [-1, -1], "city": [-1, -1], "county": [-1, -1], "street": [-1, -1],
                      "detail": [-1, -1], "person": [-1, -1], "mobile": [-1, -1]}

        for i in resutlt:
            start_index = int(i.span(0)[0] / 8)
            end_index = int((i.span(0)[1] - i.span(0)[0]) / 8) + start_index
            if i.group(0)[0:8] == "B-provin":
                shiti_dict["province"][0] = start_index
                shiti_dict["province"][1] = end_index
            if i.group(0)[0:8] == "B-cities":
                shiti_dict["city"][0] = start_index
                shiti_dict["city"][1] = end_index
            if i.group(0)[0:8] == "B-county":
                shiti_dict["county"][0] = start_index
                shiti_dict["county"][1] = end_index
            if i.group(0)[0:8] == "B-street":
                shiti_dict["street"][0] = start_index
                shiti_dict["street"][1] = end_index
            if i.group(0)[0:8] == "B-detail":
                shiti_dict["detail"][0] = start_index
                shiti_dict["detail"][1] = end_index
            if i.group(0)[0:8] == "B-person":
                shiti_dict["person"][0] = start_index
                shiti_dict["person"][1] = end_index
            if i.group(0)[0:8] == "B-mobile":
                shiti_dict["mobile"][0] = start_index
                shiti_dict["mobile"][1] = end_index
        if shiti_dict["province"][0] != -1:
            pro = "".join(text_list[shiti_dict["province"][0]:shiti_dict["province"][1]])
            result_word["province"].append(pro)
        if shiti_dict["city"][0] != -1:
            cit = "".join(text_list[shiti_dict["city"][0]:shiti_dict["city"][1]])
            result_word["city"].append(cit)
        if shiti_dict["county"][0] != -1:
            cou = "".join(text_list[shiti_dict["county"][0]:shiti_dict["county"][1]])
            result_word["county"].append(cou)
        if shiti_dict["street"][0] != -1:
            stre = "".join(text_list[shiti_dict["street"][0]:shiti_dict["street"][1]])
            result_word["street"].append(stre)
        if shiti_dict["detail"][0] != -1:
            det = "".join(text_list[shiti_dict["detail"][0]:shiti_dict["detail"][1]])
            result_word["detail"].append(det)
        if shiti_dict["person"][0] != -1:
            per = "".join(text_list[shiti_dict["person"][0]:shiti_dict["person"][1]])
            result_word["person"].append(per)
        if shiti_dict["mobile"][0] != -1:
            mob = "".join(text_list[shiti_dict["mobile"][0]:shiti_dict["mobile"][1]])
            res = re.compile('(13\d{9}|14[5|7]\d{8}|15\d{9}|166{\d{8}|17[3|6|7]{\d{8}|18\d{9})')
            s = re.findall(res, curr_text)
            try:
                mob = s[0]
            except:
                mob = mob[0:11]
            result_word["cellphones"].append(mob)
        print("".join(text_list))
        print(result_word)


dataframe = pd.read_csv("data/squence-label/dev.csv")
text_list = dataframe["text"]
label = dataframe["label"]

for idx, text in enumerate(text_list):
    predict(text)