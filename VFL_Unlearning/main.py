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
from dataset import get_dataset
from copy import deepcopy
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(f"VFU.{__name__}")



if __name__ == '__main__':
    args = config()
    # update_logger(args)
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    vfl = VFL(args)
    start_time = time.time()
    for i in range(args.epochs):
        vfl.train()
        print(f"Epoch: {i}, Train loss: {vfl.train_loss_meter.get()}, "
                    f"Train acc: {vfl.train_acc_meter.get()}")
        logger.info(f"Epoch: {i}, Train loss: {vfl.train_loss_meter.get()}, "
                    f"Train acc: {vfl.train_acc_meter.get()}")
        vfl.test()
        print(f"Epoch: {i}, Test acc: {vfl.test_acc_meter.get()}")
        logger.info(f"Epoch: {i}, Test acc: {vfl.test_acc_meter.get()}")

        vfl.adjust_lr()
    logger.info(f"Round: Best, Test acc: {vfl.best_test_acc}")
    logger.info(f"Time for retrain unlearning")
    end_time = time.time()
    duration = end_time - start_time
    duration = round(duration,2)
    logger.info("--- %s seconds ---" % (duration))
    logger.info("\n")
    vfl.save()


    
