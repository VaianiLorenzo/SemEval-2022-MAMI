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

# hold out: percentage of elements in train set
percentage_train = 0.75

# batch size: adjust based on GPU memory
batch_size = 64

# BERT tokenizer for text
text_model_name = "bert-base-cased"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

# extracts data from CSV file
df = pd.read_csv("TRAINING/training.csv", sep="\t")
df = df.sample(frac=1)
names = list(df["file_name"])
misogynous = list(df['misogynous'])
shaming = list(df['shaming'])
stereotype = list(df['stereotype'])
objectification = list(df['objectification'])
violence = list(df['violence'])
text = list(df["Text Transcription"])

####################
# TRAIN DATALOADER #
####################

# train lists
train_image_path = []
train_text = []
train_label = []

for i in tqdm(range(int(len(names)*percentage_train))):
    train_image_path.append("TRAINING/" + names[i])
    train_text.append(text[i])
    train_label.append(misogynous[i])

train_dataloader = MAMI_binary_dataset(train_text, train_image_path, text_tokenizer, train_label, max_length=128)
train_dataloader = DataLoader(train_dataloader, batch_size=batch_size, shuffle=True,
                              num_workers=24, pin_memory=True, collate_fn=collate_fn, prefetch_factor=4)
torch.save(train_dataloader, "dataloaders/train_binary_dataloader.bkp")
del train_dataloader
gc.collect()


####################
# VAL DATALOADER #
####################

# val lists
val_image_path = []
val_text = []
val_label = []

for i in tqdm(range(int(len(names)*percentage_train), len(names), 1)):
    val_image_path.append("TRAINING/" + names[i])
    val_text.append(text[i])
    val_label.append(misogynous[i])

val_dataloader = MAMI_binary_dataset(val_text, val_image_path, text_tokenizer, val_label, max_length = 128)
val_dataloader = DataLoader(val_dataloader, batch_size=batch_size, shuffle=True,
                            num_workers=24, pin_memory=True, collate_fn=collate_fn, prefetch_factor=4)
torch.save(val_dataloader, "dataloaders/val_binary_dataloader.bkp")
