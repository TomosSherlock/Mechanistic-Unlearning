import argparse
import collections
import copy
import json
import logging
import math
import os
import random
import re
import signal
import ssl
import time
import urllib.request
import sys
from datetime import datetime
from os import path as osp
from tqdm import tqdm
#from solver import solve_isotropic_covariance
from typing import overload

import numpy as np

# Blind torch
try:
    import torch
    import torchvision
    import torch.distributions as distributions
except ImportError:
    torch = None
    torchvision = None
    distributions = None

logger = logging.getLogger(__name__)



class LoggerPrecisionFilter(logging.Filter):
    def __init__(self, precision):
        super().__init__()
        self.print_precision = precision

    def str_round(self, match_res):
        return str(round(eval(match_res.group()), self.print_precision))

    def filter(self, record):
        # use regex to find float numbers and round them to specified precision
        if not isinstance(record.msg, str):
            record.msg = str(record.msg)
        if record.msg != "":
            if re.search(r"([-+]?\d+\.\d+)", record.msg):
                record.msg = re.sub(r"([-+]?\d+\.\d+)", self.str_round,
                                    record.msg)
        return True
    



def update_logger(args, clear_before_add=True):
    import os
    import logging

    root_logger = logging.getLogger("VFU")

    # clear all existing handlers and add the default stream
    if clear_before_add:
        root_logger.handlers = []
        handler = logging.StreamHandler()
        logging_fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(" \
                      "message)s"
        handler.setFormatter(logging.Formatter(logging_fmt))

        root_logger.addHandler(handler)

    # update level
    if args.verbose > 0:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARN
        logger.warning("Skip DEBUG/INFO messages")
    root_logger.setLevel(logging_level)

    if args.outdir == "":
        args.outdir = os.path.join(os.getcwd(), "exp")
    if args.expname == "":
        args.expname = f"{args.unlearn_method}_{args.model_type}_on" \
                      f"_{args.data}_lr{args.optimizer_lr}"
    args.expname = args.expname.replace(" ", "")
    args.expname = f"{args.expname}"
    args.outdir = os.path.join(args.outdir, args.expname)
    outdir = os.path.join(args.outdir, "_sub_exp" +
                          datetime.now().strftime('_%Y%m%d%H%M%S')
                          ) 
    #outdir = os.path.join(outdir, (args.std)
    while os.path.exists(outdir):
        time.sleep(1)
        outdir = os.path.join(
            args.outdir,
            "sub_exp" + datetime.now().strftime('_%Y%m%d%H%M%S'))
    args.outdir = outdir
    # if not, make directory with given name
    os.makedirs(args.outdir, exist_ok=True)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(args.outdir, 'exp_print.log'))
    fh.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter(
        "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    fh.setFormatter(logger_formatter)
    root_logger.addHandler(fh)

    # set print precision for terse logging
    np.set_printoptions(precision=args.print_decimal_digits)
    precision_filter = LoggerPrecisionFilter(args.print_decimal_digits)
    # attach the filter to the fh handler to propagate the filter, since
    # "Filters, unlike levels and handlers, do not propagate",
    # ref https://stackoverflow.com/questions/6850798/why-doesnt-filter-
    # attached-to-the-root-logger-propagate-to-descendant-loggers
    for handler in root_logger.handlers:
        handler.addFilter(precision_filter)

    import socket
    root_logger.info(f"the current machine is at"
                     f" {socket.gethostbyname(socket.gethostname())}")
    root_logger.info(f"the current dir is {os.getcwd()}")
    root_logger.info(f"the output dir is {args.outdir}")

    root_logger.info("the used configs are: \n" + str(args))



# def save_args(args):
#     """
#         1) make the cfg attributes immutable;
#         2) save the frozen cfg_check_funcs into
#         "self.outdir/config.yaml" for better reproducibility;
#         3) if self.wandb.use=True, update the frozen config

#     :return:
#     """
#     # save the final cfg
#     with open(os.path.join(args.outdir, "config.yaml"), 'w') as outfile:
#         from contextlib import redirect_stdout
#         with redirect_stdout(outfile):
#             tmp_cfg = copy.deepcopy(args)
#             print(tmp_cfg.dump())



class AvgMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.n = 0
        self.avg = 0.

    def update(self, val, n):
        assert n > 0
        self.val += val * n
        self.n += n
        self.avg = self.val / self.n

    def get(self):
        return self.avg
    
def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write("\033[K" + bar + "\r\n")

    sys.stdout.flush()

def evaluate_non_federated(model, data_loader, args):
    model.eval()
    test_acc = AvgMeter()
    test_acc.reset()
    with torch.no_grad():
        for step, (batch_x, batch_y) in tqdm(enumerate(data_loader)):
            if args.data != 'yahoo':
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
                x = batch_x
            elif args.data == 'custom-1':
                    x = torch.stack([batch_x])
                    x = torch.permute(x, (1,0,2,3))
            else:
                x = batch_x
            
            output = model(x)
            acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
            test_acc.update(acc.item(), batch_y.size(0))
    
        return test_acc.get()

def evaluate(bottom_model_A, bottom_model_B, top_model, data_loader, args):
    # if the top model is None, we are not using federated learning
    if top_model == None:
        return evaluate_non_federated(bottom_model_A, data_loader, args)

    bottom_model_A.eval()
    bottom_model_B.eval()
    top_model.eval()
    test_acc = AvgMeter()
    test_acc.reset()
    with torch.no_grad():
        for step, (batch_x, batch_y) in tqdm(enumerate(data_loader)):
            if args.data != 'yahoo':
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            if args.data == 'cifar10':
                x_a = batch_x[:, :, :, 0:16]
                x_b = batch_x[:, :, :, 16:32]
            elif args.data == 'cifar100':
                x_a = batch_x[:, :, :, 0:16]
                x_b = batch_x[:, :, :, 16:32]
            elif args.data == 'mnist':
                x_a = batch_x[:, :, :, 0:14]
                x_b = batch_x[:, :, :, 14:28]
            elif args.data == 'mri':
                x_a = batch_x[:, :, :, 0:112]
                x_b = batch_x[:, :, :, 112:224]
            elif args.data == 'ct':
                x_a = batch_x[:, :, :, 0:112]
                x_b = batch_x[:, :, :, 112:224]
            elif args.data == 'yahoo':
                for i in range(len(batch_x)):
                    batch_x[i] = batch_x[i].long().cuda()
                batch_y = batch_y[0].long().cuda()
                x_b = batch_x[1]
                x_a = batch_x[0]
            elif args.data == 'custom':
                x_a = batch_x[:, :, :, 0:14]
                x_b = batch_x[:, :, :, 14:28]
            elif args.data == 'custom-1':
                    x_a = torch.stack([batch_x[:, :, 0:14]])
                    x_b = torch.stack([batch_x[:, :, 14:28]])
                    x_a = torch.permute(x_a, (1,0,2,3))
                    x_b = torch.permute(x_b, (1,0,2,3))
            else:
                raise ValueError(f'No dataset named {args.data}!')
            output_tensor_bottom_model_a = bottom_model_A(x_a)
            output_tensor_bottom_model_b = bottom_model_B(x_b)
            output = top_model(output_tensor_bottom_model_a, output_tensor_bottom_model_b)
            acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
            test_acc.update(acc.item(), batch_y.size(0))
    
        return test_acc.get()