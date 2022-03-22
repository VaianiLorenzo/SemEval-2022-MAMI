import pandas as pd
import torch
from tqdm import tqdm

import os

from utils.config import get_cfg_defaults

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.multiprocessing.set_sharing_strategy('file_system')

path_output_file = os.path.join("data", "submission_file.csv")

if __name__ == "__main__":
    cfg = get_cfg_defaults()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(os.path.join(cfg.TEST.DIR_CHECKPOINTS, cfg.TEST.FILE_CHECKPOINT))
    model = model.to(device)
    model.eval()

    test_dataloader = torch.load(os.path.join("data", "dataloaders", f"test_{cfg.MODEL.TYPE}_dataloader.bkp"))
    print("Test dataloader length:", len(test_dataloader))

    preds = []
    names = []
    with torch.no_grad():
        for index, data in enumerate(tqdm(test_dataloader), 0):

            if cfg.MODEL.TYPE == "base":
                texts, images, path = data
            elif cfg.MODEL.TYPE == "visual_bert":
                texts, path = data
                images = path

            predicted = model(texts, images)

            for p, n in zip(torch.sigmoid(predicted).tolist(), path):
                preds.append(int(p >= 0.5))
                names.append(n[5:])

    df = pd.DataFrame()
    df["names"] = names
    df["predictions"] = preds
    df.to_csv(path_output_file, sep="\t", header=False, index=False)
