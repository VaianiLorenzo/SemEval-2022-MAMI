import os

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = acc.item() * 100
    return acc


def train_model(cfg, model, device, n_epochs, optimizer, scheduler, train_dataloader, val_dataloader,
                path_dir_checkpoint, comet_exp):
    loss_function = BCEWithLogitsLoss()

    for epoch in range(0, n_epochs):
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        model.train()

        list_outputs = []
        ground_truth = []
        print("init. training")
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
        train_f1 = f1_score(np.array(ground_truth), torch.round(torch.sigmoid(torch.tensor(list_outputs))).numpy())

        print("LR:", scheduler.get_last_lr())
        print('Loss after epoch %5d: %.8f' % (epoch + 1, current_loss / len(train_dataloader)))
        print("Train Accuracy: ", train_acc)
        print("Train F1: ", train_f1)

        # saving as checkpoint
        if cfg.MODEL.TYPE == "base":
            file_name = f"MAMI_model_{cfg.MODEL.BASELINE_MODALITY}_{epoch}.model"
        elif cfg.MODEL.TYPE == "visual_bert":
            file_name = f"MAMI_vb_model_{cfg.MODEL.MASKR_MODALITY}_{epoch}.model"
        torch.save(model, os.path.join(path_dir_checkpoint, file_name))

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
        val_acc = binary_acc(torch.tensor(list_outputs), torch.tensor(ground_truth))
        val_f1 = f1_score(np.array(ground_truth), torch.round(torch.sigmoid(torch.tensor(list_outputs))).numpy())

        print("Validation Loss:", avg_val_loss)
        print("Validation Accuracy: ", val_acc)
        print("Validation F1: ", val_f1)

        f = open("log_file.txt", "a+")
        f.write("Epoch " + str(epoch + 1) + ":\n")
        f.write("\tTrain loss:\t\t%.8f \n" % (current_loss / len(train_dataloader)))
        f.write("\tTrain ACCURACY:\t" + str(train_acc) + "\n")
        f.write("\tTrain F1:\t" + str(train_f1) + "\n")
        f.write("\tValidation loss:\t%.8f \n" % (avg_val_loss))
        f.write("\tValidation ACCURACY:\t" + str(val_acc) + "\n")
        f.write("\tValidation F1:\t" + str(val_f1) + "\n")
        f.close()

        if cfg.COMET.ENABLED:
            comet_exp.log_metrics(
                {"Loss": current_loss / len(train_dataloader)},
                prefix="Train",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"Accuracy": train_acc},
                prefix="Train",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"F1": train_f1},
                prefix="Train",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"Loss": avg_val_loss},
                prefix="Validation",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"Accuracy": val_acc},
                prefix="Validation",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"F1": val_f1},
                prefix="Validation",
                step=(epoch + 1),
            )
