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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma = 0.1)

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

        if i % 10 == 0 and i != 0:
            print("LR:", scheduler.get_last_lr())
            print('Loss after mini-batch %5d: %.8f' %
                  (i + 1, current_loss / 10))
            current_loss = 0.0
            #print("TARGETS:", list(targets))
            #print("OUTPUTS:", outputs)
            if i % 100 == 0:
                epoch_name = "MAMI_binary_model_" + str(epoch) + "-" + str(i) + ".model"
                ckp_dir = checkpoint_dir + str(epoch_name) 
                torch.save(model, ckp_dir)


    # saving as checkpoint
    epoch_name = "MAMI_binary_model_" + str(epoch) + ".model"
    ckp_dir = checkpoint_dir + str(epoch_name)
    torch.save(model, ckp_dir)


    '''
    ##### Validation #####
    mater.eval()
    total_val_loss = 0
    counter_zero = 0
    counter_out_0 = 0
    counter_out_01 = 0
    counter_out_001 = 0
    list_outputs = []
    for i, batch in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            audios, texts, targets = data
            audios = torch.stack(audios).to(device)
            targets = torch.stack(targets).to(device)

            outputs = mater(audios, texts)
            out = [o.item() for o in outputs]
            list_outputs.extend(list(out))
            counter_zero += len([o for o in out if o == 0])
            counter_out_0 += len([o for o in out if o > 0])
            counter_out_01 += len([o for o in out if o > 0.1])
            counter_out_001 += len([o for o in out if o > 0.01])

        total_val_loss += loss_function(outputs, targets)

    # Plot Histogram on x
    x = list_outputs
    plt.hist(x, bins=50)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
    plt.savefig('histograms/'+target_score+'_val_pred_ep_'+str(epoch)+'_distribution.png')

    print("Predicted = 0", counter_zero)
    print("Predicted > 0", counter_out_0)
    print("Predicted > 0.01", counter_out_001)
    print ("Predicted > 0.1", counter_out_01)
    avg_val_loss = total_val_loss / len(val_dataloader)
    print("Validation Loss: %.8f" % (avg_val_loss))
    f = open("val_loss_"+ target_score + ".txt", "a+")
    f.write("Validation loss ("+target_score+") at epoch " + str(epoch) + ":\t %.8f \n" % (avg_val_loss))
    f.close()
    '''

# Process is complete.
print('Training process has finished.')