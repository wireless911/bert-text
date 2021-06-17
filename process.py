import pandas as pd

dataframe = pd.read_excel("data/addressInfo.xlsx")
dataframe = dataframe.where(dataframe.notnull(), None)

data = dataframe.itertuples(index=False)

data_list = []

for r in data:
    mapping = dict(
        person=r.person,
        mobile=str(int(r.mobile)) if r.mobile else None,
        province=r.province,
        city=r.city,
        county=r.county,
        street=r.street,
        detail=r.detail
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
                    start = f"B-{key}"
                    end = f"I-{key}"
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
df = pd.DataFrame(data_list, columns=["text", "label"])
df.to_csv("data/train.csv",index=False)
