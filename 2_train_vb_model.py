import numpy as np
import pandas as pd
import random
import comet_ml
from comet_ml import Experiment
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW
from torch import optim
from tqdm import tqdm
import os
from vb_model import MAMI_vb_binary_model
import logging
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

comet_log = True
if comet_log:
    experiment = Experiment(
        api_key="LiMIt9D5WsCZo294IIYymGhdv",
        project_name="mami",
        workspace="vaianilorenzo",
    )

# Arguments passed through command line
batch_size = None
n_workers = None
n_epochs = None

checkpoint_dir = "checkpoints_vb_binary_model/"

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

def train_model(device, n_epochs, lr, step_size, train_dataloader, val_dataloader):
    loss_function = BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma = 1)

    for epoch in range(0, n_epochs):  # 5 epochs at maximum
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        model.train()

        list_outputs = []
        ground_truth = []
        print ("init. training")
        for i, data in enumerate(tqdm(train_dataloader), 0):
            # Get and prepare inputs
            texts, images, targets = data
            targets = torch.tensor(targets).to(device).float()

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(texts, images)

            # update lists for accuracy computation
            out = [o.item() for o in outputs]
            list_outputs.extend(list(out))
            tar = [t.item() for t in targets]
            ground_truth.extend(tar)

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

        train_acc = binary_acc(torch.tensor(list_outputs), torch.tensor(ground_truth))

        print("LR:", scheduler.get_last_lr())
        print('Loss after epoch %5d: %.8f' % (epoch + 1, current_loss / len(train_dataloader)))
        print("Train Accuracy", train_acc)

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
        val_acc = binary_acc(torch.tensor(list_outputs),torch.tensor(ground_truth))

        print("Validation Loss:", avg_val_loss)
        print("Validation Accuracy", val_acc)

        f = open("log_file.txt", "a+")
        f.write("Epoch " + str(epoch + 1) + ":\n")
        f.write("\tTrain loss:\t\t%.8f \n" % (current_loss / len(train_dataloader)))
        f.write("\tTrain ACCURACY:\t" + str(train_acc) + "\n")
        f.write("\tValidation loss:\t%.8f \n" % (avg_val_loss))
        f.write("\tValidation ACCURACY:\t" + str(val_acc) + "\n")
        f.close()

        if comet_log:
            experiment.log_metrics(
                {"Loss": current_loss / len(train_dataloader)},
                prefix="Train",
                step=(epoch + 1),
            )

            experiment.log_metrics(
                {"Accuracy": train_acc},
                prefix="Train",
                step=(epoch + 1),
            )

            experiment.log_metrics(
                {"Loss": avg_val_loss},
                prefix="Validation",
                step=(epoch + 1),
            )

            experiment.log_metrics(
                {"Accuracy": val_acc},
                prefix="Validation",
                step=(epoch + 1),
            )

if __name__ == "__main__":
    #-------------------#
    # COMMAND LINE ARGS #
    #-------------------#
    parser = argparse.ArgumentParser(description="Running Server ML")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs",
        default=100, required=False)
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=1e-5, required=False)
    parser.add_argument(
        "--gamma",
        type=float,
        help="Gamma value for optimizer",
        default=0.5, required=False)


    args = parser.parse_args()

    n_epochs = args.epochs
    lr = args.lr
    gamma = args.gamma

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading train dataloader..")
    train_dataloader = torch.load("dataloaders/train_vb_binary_dataloader.bkp")
    print("Loading val dataloader..")
    val_dataloader = torch.load("dataloaders/val_vb_binary_dataloader.bkp")

    model = MAMI_vb_binary_model(device=device)
    model.to(device)
    model.train()

    percentage_epochs_per_step = 0.4
    step_size = n_epochs * len(train_dataloader) * percentage_epochs_per_step

    f = open("log_file.txt", "a+")
    f.write("START TRAINING - " + str(n_epochs) + " epochs - LR: " + str(lr) + " - gamma: " + str(
        gamma) + " - step_size: " + str(percentage_epochs_per_step * n_epochs) + " epochs\n")
    f.close()

    train_model(device, n_epochs, lr, step_size, train_dataloader, val_dataloader)