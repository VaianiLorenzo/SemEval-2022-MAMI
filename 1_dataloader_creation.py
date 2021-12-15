import os
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import gc
import glob
import pandas as pd
from pympler.asizeof import asizeof
import resource
import copy
from MAMI_binary_dataset import MAMI_binary_dataset
from transformers import AutoTokenizer, AutoModel

import random

# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns


# a simple custom collate function
def collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return [text, img, label]

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    torch.device('cpu')
    device = torch.device('cpu')

batch_size = 8
text_model_name = "bert-base-cased"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

df = pd.read_csv("TRAINING/training.csv", sep="\t")
names = df["file_name"]
print(names)
df = df.sample(frac=1)
names = df["file_name"]
print(names)
exit()


####################
# TRAIN DATALOADER #
####################

# train lists
train_image_path = []
train_text = []
train_label = []

with open("train_shows.txt", "r") as train_shows:
    train_files = train_shows.readlines()

train_size = len(train_files)

print("Train set:", train_size, "shows")
tot_size = 0

for f in tqdm(train_files):
    df = pd.read_feather(f[:-1])

    list_path = df["path_audio"].values
    list_start_time = df["start_time"].values
    list_end_time = df["end_time"].values
    list_text = df["sentence_text"].values
    list_labels = df[target_score].values

    del df["podcast_id"]
    del df["start_time"]
    del df["end_time"]
    del df["sentence_text"]
    del df["path_audio"]
    del df["description"]
    del df["sbert_score"]
    del df
    gc.collect()

    for i, v in enumerate(list_labels):
        train_path.append(list_path[i])
        train_start_time.append(list_start_time[i])
        train_end_time.append(list_end_time[i])
        train_text.append(list_text[i])
        train_labels.append(float(v))

    del list_path
    del list_start_time
    del list_end_time
    del list_text
    del list_labels

    gc.collect()

train_dataloader = MATeR_Dataset(train_path, train_start_time, train_end_time, train_text, torch.tensor(train_labels),
                                 audio_tokenizer, text_tokenizer)

del train_path
del train_start_time
del train_end_time
del train_text
del train_labels

train_dataloader = DataLoader(train_dataloader, batch_size=batch_size, shuffle=True,
                              num_workers=32, pin_memory=False, collate_fn=collate_fn, prefetch_factor=4)
torch.save(train_dataloader, "dataloaders/train_dataloader_" + target_score + ".bkp")

del train_dataloader
gc.collect()

####################
# VAL DATALOADER #
####################

# val lists
val_path = []
val_start_time = []
val_end_time = []
val_text = []
val_labels = []

with open("val_shows.txt", "r") as val_shows:
    val_files = val_shows.readlines()

val_size = len(val_files)

print("Val set:", val_size, "shows")
tot_size = 0

for f in tqdm(val_files):
    df = pd.read_feather(f[:-1])

    list_path = df["path_audio"].values
    list_start_time = df["start_time"].values
    list_end_time = df["end_time"].values
    list_text = df["sentence_text"].values
    list_labels = df[target_score].values

    del df["podcast_id"]
    del df["start_time"]
    del df["end_time"]
    del df["sentence_text"]
    del df["path_audio"]
    del df["description"]
    del df["sbert_score"]
    del df
    gc.collect()

    for i, v in enumerate(list_labels):
        val_path.append(list_path[i])
        val_start_time.append(list_start_time[i])
        val_end_time.append(list_end_time[i])
        val_text.append(list_text[i])
        val_labels.append(float(v))

    del list_path
    del list_start_time
    del list_end_time
    del list_text
    del list_labels

    gc.collect()

val_dataloader = MATeR_Dataset(val_path, val_start_time, val_end_time, val_text, torch.tensor(val_labels),
                               audio_tokenizer, text_tokenizer)

del val_path
del val_start_time
del val_end_time
del val_text
del val_labels

val_dataloader = DataLoader(val_dataloader, batch_size=batch_size, shuffle=True,
                            num_workers=32, pin_memory=False, collate_fn=collate_fn, prefetch_factor=4)
torch.save(val_dataloader, "dataloaders/val_dataloader_" + target_score + ".bkp")
