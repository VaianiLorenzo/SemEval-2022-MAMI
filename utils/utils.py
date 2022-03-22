import os
import glob
import argparse
import torch
from utils.config import get_cfg_defaults as get_cfg_dataloaders


def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="SemEval 2022 MAMI Solution")
    parser.add_argument(
        "--cfg",
        help="Allows to specify values for the configuration entries to override default settings",
        nargs=argparse.REMAINDER, required=False)

    args = parser.parse_args()
    return args


def read_dataloaders_config():
    cfg = get_cfg_dataloaders()

    cmd_args = parse_cmd_line_params()

    cmd_cfg = cmd_args.cfg
    if cmd_cfg is not None:
        try:
            cfg.merge_from_list(cmd_cfg)
        except:
            print("Command-line parameter parsing error")
            exit()

    cfg.freeze()

    return cfg