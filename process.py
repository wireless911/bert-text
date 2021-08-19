import pandas as pd

dataframe = pd.read_excel("data/addressInfo.xlsx")
dataframe = dataframe.where(dataframe.notnull(), None)
data = dataframe.itertuples(index=False)

data_list = []

key_mapping = {
    "person":"person",
    "mobile":"mobile",
    "city":"cities",
    "province":"provin",
    "county":"county",
    "street":"street",
    "detail":"detail"
               }


for r in data:
    mapping = dict(
        person=r.person.strip() if r.person else None,
        mobile=str(int(r.mobile)) if not pd.isnull(r.mobile) else None,
        city=r.city.strip() if r.city else None,
        province=r.province.strip() if r.province else None,
        county=r.county.strip() if r.county else None,
        street=r.street.strip() if r.street else None,
        detail=r.detail.strip() if r.detail else None
    )
    address = r.address
    if address:
        address = address.replace(" ", "")
        address = address.replace("　", "")
        text = address
        tags = []

        log = 0
        for k, x in enumerate(text):
            change = False
            if k < log:
                continue
            for key, a in mapping.items():
                if a is None:
                    continue
                elif text[k:(k + len(a))] == a:
                    start = f"B-{key_mapping[key]}"
                    end = f"I-{key_mapping[key]}"
                    arr = [start] + [end] * (len(a) - 1)
                    tags.extend(arr)
                    log = k + len(a)
                    change = True
                    break
                else:
                    continue
            if not change:
                tags.append("O")

        text = " ".join([x for x in text if x != " "])
        label = " ".join(tags)
        res = (text, label)
        data_list.append(res)
import random
random.shuffle(data_list)
df = pd.DataFrame(data_list, columns=["text", "label"])
df.to_csv("data/train.csv",index=False)
