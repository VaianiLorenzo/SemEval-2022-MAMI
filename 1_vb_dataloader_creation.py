import os
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import gc
import glob
import pandas as pd
from pympler.asizeof import asizeof
import copy
from MAMI_vb_binary_dataset import MAMI_vb_binary_dataset
from transformers import AutoTokenizer
import argparse

percentage_train = 0.75
path_dataset = "TRAINING/training.csv"

# Always use the same train/validation split
random_state = 1995

# Set through command line arguments
batch_size = None
n_workers = None

def collate_fn(self, batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return [text, img, label]

def load_data():
    df = pd.read_csv(path_dataset, sep="\t")
    df = df.sample(frac=1, random_state=random_state)
    names = list(df["file_name"])
    misogynous = list(df['misogynous'])
    shaming = list(df['shaming'])
    stereotype = list(df['stereotype'])
    objectification = list(df['objectification'])
    violence = list(df['violence'])
    text = list(df["Text Transcription"])

    return names, text, misogynous


if __name__ == "__main__":
    #-------------------#
    # COMMAND LINE ARGS #
    #-------------------#
    parser = argparse.ArgumentParser(description="Running Server ML")
    parser.add_argument(
        "--bs",
        type=int,
        help="Batch size",
        default=16, required=False)
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers for dataloaders",
        default=4, required=False)
    
    args = parser.parse_args()
    
    batch_size = args.bs
    n_workers = args.workers
    
    names, text, misogynous = load_data()
    
    # BERT tokenizer for text
    text_model_name = "bert-base-uncased"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

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

    train_dataloader = MAMI_vb_binary_dataset(train_text, train_image_path, text_tokenizer, train_label, max_length=128)
    train_dataloader = DataLoader(train_dataloader, batch_size=batch_size, shuffle=True,
                                  num_workers=n_workers, pin_memory=True, collate_fn=collate_fn, prefetch_factor=4)
    torch.save(train_dataloader, "dataloaders/train_vb_binary_dataloader.bkp")
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

    val_dataloader = MAMI_vb_binary_dataset(val_text, val_image_path, text_tokenizer, val_label, max_length = 128)
    val_dataloader = DataLoader(val_dataloader, batch_size=batch_size, shuffle=True,
                                num_workers=n_workers, pin_memory=True, collate_fn=collate_fn, prefetch_factor=4)
    torch.save(val_dataloader, "dataloaders/val_vb_binary_dataloader.bkp")