import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW
from torch import optim
from tqdm import tqdm
import os
from model import MAMI_binary_model
import logging
import multiprocessing
#torch.multiprocessing.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

# a simple custom collate function, just to show the idea
def collate_fn(batch):
    text  = [item[0] for item in batch]
    img   = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return [text, img, label]

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = acc.item() * 100
    return acc

checkpoint_dir = "checkpoints_binary_model/"

print("Loading train dataloader..")
train_dataloader = torch.load("dataloaders/train_binary_dataloader.bkp")
print("Loading val dataloader..")
val_dataloader = torch.load("dataloaders/val_binary_dataloader.bkp")

n_epochs = 5
lr = 1e-5
step_size = n_epochs * len(train_dataloader) * 0.2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# Initialize the MLP
model = MAMI_binary_model(device=device)
model.to(device)
model.train()

# Define the loss function and optimizer
#loss_function = CrossEntropyLoss()
loss_function = BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma = 1)

for epoch in range(0, n_epochs):  # 5 epochs at maximum


    print(f'Starting epoch {epoch + 1}')

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    model.train()

    print ("init. training")
    for i, data in enumerate(tqdm(train_dataloader), 0):
        # Get and prepare inputs
        texts, images, targets = data
        targets = torch.tensor(targets).to(device).float()

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = model(texts, images)

        # compute loss
        loss = loss_function(outputs, targets)

        # Perform backward pass
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # Perform optimization
        optimizer.step()
        scheduler.step()

        # Print statistics
        current_loss += loss.item()

    print("LR:", scheduler.get_last_lr())
    print('Loss after epoch %5d: %.8f' % (epoch + 1, current_loss / len(train_dataloader)))

    # saving as checkpoint
    epoch_name = "MAMI_binary_model_" + str(epoch) + ".model"
    ckp_dir = checkpoint_dir + str(epoch_name)
    torch.save(model, ckp_dir)


    ##### Validation #####
    model.eval()
    total_val_loss = 0

    list_outputs = []
    ground_truth = []
    for i, data in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            texts, images, targets = data
            targets = torch.tensor(targets).to(device).float()
            outputs = model(texts, images)
            out = [o.item() for o in outputs]
            list_outputs.extend(list(out))
            tar = [t.item() for t in targets]
            ground_truth.extend(tar)

        total_val_loss += loss_function(outputs, targets).item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    acc = binary_acc(torch.tensor(list_outputs),torch.tensor(ground_truth))

    print("Validation Loss:", total_val_loss)
    print("Validation Accuracy", acc)

    '''
    f = open("val_loss_"+ target_score + ".txt", "a+")
    f.write("Validation loss ("+target_score+") at epoch " + str(epoch) + ":\t %.8f \n" % (avg_val_loss))
    f.close()
    '''

# Process is complete.
print('Training process has finished.')