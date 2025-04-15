import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

df = pd.read_json('dataset/dataset.json', orient='index').reset_index()[:-5]

main_vocab = ["ح", "چ", "ج", "ث", "ت", "پ", "ب", "آ", "ا", "ش", "س", "ژ", "ز", "ر", "ذ", "د", "خ", "ق", "ف", "غ", "ع",
              "ظ", "ط", "ض", "ص", "ی", "ه", "و", "ن", "م", "ل", "گ", "ک"]
text = " ".join(df["cleaned_tweet"].values.tolist())
vocab = list(sorted(set(text)))

for v in main_vocab:
    if v not in vocab:
        print("v", v)

print(len(main_vocab), len(vocab))
print(vocab)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = train_df[["path", "cleaned_tweet"]]
train_df = train_df.reset_index(drop=True)

test_df = test_df[["path", "cleaned_tweet"]]
test_df = test_df.reset_index(drop=True)

print(train_df.shape)
print(test_df.shape)

save_path = "dataset/csv"
print(save_path)

train_df.to_csv(f"{save_path}/train.csv", sep=",", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep=",", encoding="utf-8", index=False)

print(train_df.shape)
print(test_df.shape)

common_voice_train = load_dataset("csv", data_files={"train": "dataset/csv/train.csv"}, delimiter=",")["train"]
common_voice_test = load_dataset("csv", data_files={"test": "dataset/csv/test.csv"}, delimiter=",")["test"]

print(common_voice_train)
print(common_voice_test)


def extract_all_chars(batch):
    all_text = " ".join(batch["cleaned_tweet"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                     remove_columns=common_voice_train.column_names)
vocab_test = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                    remove_columns=common_voice_test.column_names)


vocab_list = list(sorted(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0])))
vocab_list = [vocab for vocab in vocab_list if vocab not in [" ", "\u0307"]]
print(len(vocab_list))
print(vocab_list)

special_vocab = ["<pad>", "<s>", "</s>", "<unk>", "|"]
vocab_dict = {v: k for k, v in enumerate(special_vocab + vocab_list)}
print(len(vocab_dict))
print(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)