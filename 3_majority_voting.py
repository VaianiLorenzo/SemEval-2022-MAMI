import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import json

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.multiprocessing.set_sharing_strategy('file_system')

def test_collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    return [text, img]

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    torch.device('cpu')
    device = torch.device('cpu')

test_dataloader = torch.load("dataloaders/test_vb_binary_dataloader.bkp")
print("Test dataloader length:", len(test_dataloader))

model_list = ["MAMI_vb_binary_model_both_10.model", "MAMI_vb_binary_model_coco_24.model", "MAMI_vb_binary_model_coco_15.model"]
weights = [81.76, 81.07, 82.27]
model_dir = "checkpoints_vb_binary_model_cls"

voting_dict = {}
for m,w in zip(model_list, weights):
    model = torch.load(model_dir + "/" + m)
    model = model.to(device)
    model.eval()

    df = pd.DataFrame()
    preds = []
    names = []

    with torch.no_grad():
        for index, data in enumerate(tqdm(test_dataloader), 0):

            texts, path = data
            predicted = model(texts, path)
            for p,n in zip(torch.sigmoid(predicted).tolist(), path):
                if n not in voting_dict.keys():
                    voting_dict[n] = []
                voting_dict[n].append(p * w / sum(weights))

for k in voting_dict.keys():
    names.append(k[5:])
    if sum(voting_dict[k]) > 0.5:
        preds.append(1)
    else:
        preds.append(0)

df["names"] = names
df["predictions"] = preds
df.to_csv("submission_file.csv", sep="\t", header = False, index = False)
