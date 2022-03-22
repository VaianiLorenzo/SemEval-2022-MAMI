import os
import gc

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets.MAMI_binary_dataset import MAMI_binary_dataset
from datasets.MAMI_test_binary_dataset import MAMI_test_binary_dataset
from datasets.MAMI_test_vb_binary_dataset import MAMI_test_vb_binary_dataset
from datasets.MAMI_vb_binary_dataset import MAMI_vb_binary_dataset
from utils.collate_functions import base_collate_fn, base_test_collate_fn, vb_collate_fn, vb_test_collate_fn
from utils.utils import read_dataloaders_config

path_output_dir = os.path.join("data", "dataloaders")
path_train_dataset = os.path.join("data", "TRAINING", "training.csv")
path_test_dataset = os.path.join("data", "test", "test.csv")

# Always use the same train/validation split
random_state = 1995


def load_data(path_dataset):
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
    cfg = read_dataloaders_config()

    # Create output directory if does not exist
    if not os.path.isdir(path_output_dir):
        os.mkdir(path_output_dir)

    names, text, misogynous = load_data(path_train_dataset)

    if cfg.MODEL.TYPE == "base":
        text_model_name = "bert-base-cased"
        collate_fn = base_collate_fn
        test_collate_fn = base_test_collate_fn
    elif cfg.MODEL.TYPE == "visual_bert":
        text_model_name = "bert-base-uncased"
        collate_fn = vb_collate_fn
        test_collate_fn = vb_test_collate_fn

    # BERT tokenizer for text
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    ####################
    # TRAIN DATALOADER #
    ####################
    print("Creating train dataloader...")

    # train lists
    train_image_path = []
    train_text = []
    train_label = []

    for i in tqdm(range(int(len(names) * cfg.DATALOADER.PERCENTAGE_TRAIN))):
        train_image_path.append(os.path.join("data", "TRAINING", names[i]))
        train_text.append(text[i])
        train_label.append(misogynous[i])

    if cfg.MODEL.TYPE == "base":
        train_dataloader = MAMI_binary_dataset(train_text, train_image_path, text_tokenizer, train_label,
                                               max_length=128)
    elif cfg.MODEL.TYPE == "visual_bert":
        train_dataloader = MAMI_vb_binary_dataset(train_text, train_image_path, text_tokenizer, train_label,
                                                  max_length=128)
    train_dataloader = DataLoader(train_dataloader, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=collate_fn,
                                  prefetch_factor=4)
    torch.save(train_dataloader, os.path.join(path_output_dir, f"train_{cfg.MODEL.TYPE}_dataloader.bkp"))
    del train_dataloader
    gc.collect()

    ####################
    # VAL DATALOADER #
    ####################
    print("Creating validation dataloader...")

    # val lists
    val_image_path = []
    val_text = []
    val_label = []

    for i in tqdm(range(int(len(names) * cfg.DATALOADER.PERCENTAGE_TRAIN), len(names), 1)):
        val_image_path.append(os.path.join("data", "TRAINING", names[i]))
        val_text.append(text[i])
        val_label.append(misogynous[i])

    if cfg.MODEL.TYPE == "base":
        val_dataloader = MAMI_binary_dataset(train_text, train_image_path, text_tokenizer, train_label,
                                             max_length=128)
    elif cfg.MODEL.TYPE == "visual_bert":
        val_dataloader = MAMI_vb_binary_dataset(train_text, train_image_path, text_tokenizer, train_label,
                                                max_length=128)
    val_dataloader = DataLoader(val_dataloader, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=collate_fn,
                                prefetch_factor=4)
    torch.save(val_dataloader, os.path.join(path_output_dir, f"val_{cfg.MODEL.TYPE}_dataloader.bkp"))

    ####################
    # TEST DATALOADER #
    ####################
    print("Creating test dataloader...")

    df = pd.read_csv(path_test_dataset, sep="\t")
    images = list(df["file_name"])
    for i in range(len(images)):
        images[i] = os.path.join("data", "test", images[i])
    texts = list(df["Text Transcription"])

    if cfg.MODEL.TYPE == "base":
        test_dataloader = MAMI_test_binary_dataset(texts, images, text_tokenizer, max_length=128)
    elif cfg.MODEL.TYPE == "visual_bert":
        test_dataloader = MAMI_test_vb_binary_dataset(texts, images, text_tokenizer, max_length=128)
    test_dataloader = DataLoader(test_dataloader, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                 num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=test_collate_fn,
                                 prefetch_factor=4)
    torch.save(test_dataloader, os.path.join(path_output_dir, f"test_{cfg.MODEL.TYPE}_dataloader.bkp"))
