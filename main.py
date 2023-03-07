import os, sys

import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV4-PYTORCH")
    parser.add_argument("--gpus", type = int, nargs = "+", default = [], help = "List of device ids")
    parser.add_argument("--mode", dest = "mode", help = "train / eval / test",
                        default = None, type = str)
    parser.add_argument("--cfg", dest = "cfg", help = "the path of model config",
                        default = None, type = str)
    parser.add_argument("--checkpoint", dest = "checkpoint", help = "the path of checkpoint",
                        default = None, type = str)
    parser.add_argument("--pretrained", dest = "pretrained", help = "the path of pre-trained model(weight)",
                        default = None, type = str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def train(cfg_params = None, using_gpus = None):
    transforms = get_Transforms(cfg_params, is_train = True)
    #Train dataloader
    train_data = YOLOdata(is_train = True,
                          transform = transforms,
                          cfg_params = cfg_params)

