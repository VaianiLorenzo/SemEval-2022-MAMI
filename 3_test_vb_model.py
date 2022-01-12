import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import json

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.multiprocessing.set_sharing_strategy('file_system')


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    torch.device('cpu')
    device = torch.device('cpu')

modality = "both"
epoch = 10

model_dir = "checkpoints_vb_binary_model"
model_name = "MAMI_vb_binary_model"
model = torch.load(model_dir + "/" + model_name + "_" + modality + "_" + str(epoch) + ".model")
model = model.to(device)
model.eval()

def test_collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    return [text, img]


test_dataloader = torch.load("dataloaders/test_vb_binary_dataloader.bkp")
print("Test dataloader length:", len(test_dataloader))

df = pd.DataFrame()
preds = []
names = []

with torch.no_grad():
    for index, data in enumerate(tqdm(test_dataloader), 0):

        texts, path = data

        predicted = model(texts, path)

        for p,n in zip(torch.sigmoid(predicted).tolist(), path):
            preds.append(int(p>=0.5))
            names.append(n[5:])


df["names"] = names
df["predictions"] = preds
df.to_csv("submission_file.csv", sep="\t", header = False, index = False)
