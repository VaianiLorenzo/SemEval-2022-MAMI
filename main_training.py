import logging
import os

import torch
from comet_ml import Experiment

from models.model import MAMI_binary_model
from models.vb_model import MAMI_vb_binary_model
from utils.config import get_cfg_defaults
from utils.training import train_model


torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

path_output_dir = os.path.join("data", "dataloaders")

if __name__ == "__main__":
    cfg = get_cfg_defaults()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    experiment = None
    if cfg.COMET.ENABLED:
        experiment = Experiment(
            api_key=cfg.COMET.API_KEY,
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
        )

    print("Loading train dataloader..")
    train_dataloader = torch.load(os.path.join(path_output_dir, f"train_{cfg.MODEL.TYPE}_dataloader.bkp"))
    print("Loading val dataloader..")
    val_dataloader = torch.load(os.path.join(path_output_dir, f"val_{cfg.MODEL.TYPE}_dataloader.bkp"))

    if cfg.MODEL.TYPE == "base":
        model = MAMI_binary_model(device=device, modality=cfg.MODEL.BASELINE_MODALITY)
    elif cfg.MODEL.TYPE == "visual_bert":
        model = MAMI_vb_binary_model(device=device, class_modality=cfg.MODEL.CLASS_MODALITY,
                                     maskr_modality=cfg.MODEL.MASKR_MODALITY)
    # Create checkpoint directory if it does not exist
    path_dir_checkpoint = os.path.join("data", f"checkpoints_{cfg.MODEL.TYPE}")
    if not os.path.isdir(path_dir_checkpoint):
        os.mkdir(path_dir_checkpoint)
    model.to(device)

    # Configure optimizer
    percentage_epochs_per_step = 0.4
    step_size = cfg.TRAINING.EPOCHS * len(train_dataloader) * percentage_epochs_per_step
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAINING.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=cfg.TRAINING.GAMMA)

    # Init training log file
    with open("log_file.txt", "a+") as f:
        f.write(
            "START TRAINING - " + str(cfg.TRAINING.EPOCHS) + " epochs - LR: " + str(cfg.TRAINING.LR) + " - gamma: " + str(
                cfg.TRAINING.GAMMA) + " - step_size: " + str(
                percentage_epochs_per_step * cfg.TRAINING.EPOCHS) + " epochs\n")

    train_model(cfg=cfg, model=model, device=device, n_epochs=cfg.TRAINING.EPOCHS, optimizer=optimizer,
                scheduler=scheduler, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                path_dir_checkpoint=path_dir_checkpoint, comet_exp=experiment)
