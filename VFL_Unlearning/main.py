# -*- coding: utf-8 -*-
# ---
# @File: main.py
# @Author: Tae Hong Xi
# @Institution: Universiti Malaya
# @E-mail: taehongxi55@gmail.com
# @Time: 2024/7/10
# ---

import logging
import os

import torch
import time

from configs import config
from utils import update_logger
from vfl_framework import VFL
from model_framework import MODEL
from dataset import get_dataset
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from datetime import datetime

logger = logging.getLogger(f"VFU.{__name__}")

import wandb

# Initiate logger
def init_run(args):
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    data_file = args.data_path.split("/")[-1]
    n_biased_classes, rho = data_file.split("-")

    if (args.mode == "full"):
        unlearn_goal = ''
    elif args.unlearn_bias == 1:
        unlearn_goal = '-samples'
    else:
        unlearn_goal = '-classes'
    
    # set experiment name if not specified
    if args.vfl == 0:
        model_n = f'Resnet'
    else:
        model_n = f'Resnet_VFL-{args.vfl}'

    if args.data == 'custom':
        dataset = 'Biased_MNIST'
    else:
        dataset = args.data
 
    date = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')

    if args.expname == '---':
        args.expname = f'{model_n}-{dataset}-{n_biased_classes}-{rho}-{args.mode}{unlearn_goal}@{date}'
    

    # Start a new wandb run to track this.
    run = wandb.init(
        name=args.expname,
        entity="tomos-sherlock-mechanistic-unlearning",
        mode="offline",
        project="vfl-unlearning",
        # Track hyperparameters and run metadata.
        config={
            "architecture": "VFL Resnet",
            "active parties": 2,
            "dataset": "Biased-MNIST",
            "biased classes": n_biased_classes,
            "bias probability": rho,
            "epochs": args.epochs,
            "mode": args.mode,
            "unlearn goal" : unlearn_goal,
        },
    )

    return run
    

if __name__ == '__main__':
    args = config()

    run = init_run(args)

    # 1. Check if a GPU is available
    if torch.cuda.is_available():
        # 2. Get the index of the current GPU being used (default is 0)
        current_device = torch.cuda.current_device()
        print(f"Current GPU Index: {current_device}")

        # 3. Get the name of that specific GPU
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"Current GPU Name: {gpu_name}")  
    else:
        print("PyTorch is using the CPU.")

    if args.vfl == 0:
        vfl = MODEL(args)
    else :
        vfl = VFL(args)

    start_time = time.time()
    for i in range(args.epochs):
        run.log({"Epoch":i})
        vfl.train()
        print(f"Epoch: {i}, Train loss: {vfl.train_loss_meter.get()}, "
                    f"Train acc: {vfl.train_acc_meter.get()}")
        run.log({"Train loss" : vfl.train_loss_meter.get(), "Train acc" : vfl.train_acc_meter.get()})
        for j in range(10):
            print(f"{j}: {vfl.pc_train_acc_meter[j].get()}")
            run.log({f"Class {j} Train acc" : vfl.pc_train_acc_meter[j].get()})
        vfl.test()
        print(f"Epoch: {i}, Test acc: {vfl.test_acc_meter.get()}")
        run.log({"Test acc" : vfl.test_acc_meter.get()})
        for j in range(10):
            print(f"{j}: {vfl.pc_test_acc_meter[j].get()}")
            run.log({f"Class {j} Test acc" : vfl.pc_test_acc_meter[j].get()})

        vfl.adjust_lr()
    run.log({"Best Test acc" : vfl.best_test_acc})
    end_time = time.time()
    duration = end_time - start_time
    duration = round(duration,2)
    run.log({"Training Time (s)" : duration})
    vfl.save()


    
