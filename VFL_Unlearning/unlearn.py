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

import unlearn

from configs import config
from utils import update_logger, evaluate
from vfl_framework import VFL
from dataset import get_dataset
from copy import deepcopy
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torch import nn
from MIA import MIA
import time
import numpy as np
import random
from model import read_data_text
from fisher_forgetting import fisher_forgetting
logger = logging.getLogger(f"VFU.{__name__}")
import wandb
from datetime import datetime
    
# Initiate logger
def init_run(args):
    model_file = args.data_path.split("/")[-1]
    n_biased_classes, rho = model_file.split("-")
    if  args.unlearn_bias == 1:
        unlearn_goal = 'samples'
    else:
        unlearn_goal = 'classes'

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
        args.expname = f'Unlearn-{unlearn_goal}-{args.unlearn_method}-{model_n}-{dataset}-{n_biased_classes}-{rho}@{date}'

    # Start a new wandb run to track this.
    run = wandb.init(
        name=args.expname,
        entity="tomos-sherlock-mechanistic-unlearning",
        mode="offline",
        project="vfl-unlearning",
        # Track hyperparameters and run metadata.
        config={
            "unlearning method": args.unlearn_method,
            "biased classes": n_biased_classes,
            "bias probability": rho,
            "epochs": args.epochs,
            "mode": args.mode,
            "unlearn goal" : unlearn_goal,
        },
    )

    return run


def main():
    args = config()
    update_logger(args)
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic=True

    run = init_run(args)
    
    if args.model_type == 'resnet18':
        if args.vfl == 0:
            full_bottom_model_A = torch.load(f"{args.model_path}/full_standard_model.pt", weights_only=False)
            full_bottom_model_B = None
            full_top_model = None
            retrain_bottom_model_A = torch.load(f"{args.model_path}/retrain_standard_model.pt", weights_only=False)
            retrain_bottom_model_B = None 
            retrain_top_model = None

        else:

            if args.data=='cifar10':
                full_bottom_model_A = torch.load("/scratch/taehongxi/exp/emb_exp_resnet18_on_cifar10_lr0.001/full_train/full_bottom_model_A.pt", weights_only=False)
                full_bottom_model_B = torch.load("/scratch/taehongxi/exp/emb_exp_resnet18_on_cifar10_lr0.001/full_train/full_bottom_model_B.pt", weights_only=False)
                full_top_model = torch.load("/scratch/taehongxi/exp/emb_exp_resnet18_on_cifar10_lr0.001/full_train/full_top_model.pt", weights_only=False)
                retrain_bottom_model_A = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar10_lr0.001/retrain3/retrain_bottom_model_A.pt", weights_only=False)
                retrain_bottom_model_B = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar10_lr0.001/retrain3/retrain_bottom_model_B.pt", weights_only=False)
                retrain_top_model = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar10_lr0.001/retrain3/retrain_top_model.pt", weights_only=False)
                
            elif args.data == 'mnist':
                full_bottom_model_A = torch.load("/scratch/taehongxi/exp/EE_resnet18_on_mnist_lr0.001/full_train/full_bottom_model_A.pt", weights_only=False)
                full_bottom_model_B = torch.load("/scratch/taehongxi/exp/EE_resnet18_on_mnist_lr0.001/full_train/full_bottom_model_B.pt", weights_only=False)
                full_top_model = torch.load("/scratch/taehongxi/exp/EE_resnet18_on_mnist_lr0.001/full_train/full_top_model.pt", weights_only=False)
                retrain_bottom_model_A = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_mnist_lr0.001/retrain3/retrain_bottom_model_A.pt", weights_only=False)
                retrain_bottom_model_B = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_mnist_lr0.001/retrain3/retrain_bottom_model_B.pt", weights_only=False)
                retrain_top_model = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_mnist_lr0.001/retrain3/retrain_top_model.pt", weights_only=False)
                
            elif args.data == 'cifar100':
                full_bottom_model_A = torch.load("/scratch/taehongxi/exp/EE_resnet18_on_cifar100_lr0.001/full_train/full_bottom_model_A.pt", weights_only=False)
                full_bottom_model_B = torch.load("/scratch/taehongxi/exp/EE_resnet18_on_cifar100_lr0.001/full_train/full_bottom_model_B.pt", weights_only=False)
                full_top_model = torch.load("/scratch/taehongxi/exp/EE_resnet18_on_cifar100_lr0.001/full_train/full_top_model.pt", weights_only=False)
                retrain_bottom_model_A = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar100_lr0.001/retrain3/retrain_bottom_model_A.pt", weights_only=False)
                retrain_bottom_model_B = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar100_lr0.001/retrain3/retrain_bottom_model_B.pt", weights_only=False)
                retrain_top_model = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar100_lr0.001/retrain3/retrain_top_model.pt", weights_only=False)

            elif args.data == "mri":
                full_bottom_model_A = torch.load("/scratch/taehongxi/exp/_sub_exp_20241115165829 (1)/full_bottom_model_A.pt", weights_only=False)
                full_bottom_model_B = torch.load("/scratch/taehongxi/exp/_sub_exp_20241115165829 (1)/full_bottom_model_B.pt", weights_only=False)
                full_top_model = torch.load("/scratch/taehongxi/exp/_sub_exp_20241115165829 (1)/full_top_model.pt", weights_only=False)
                retrain_bottom_model_A = torch.load("/home/user/bryanthx/exp/EE_resnet18_on_mri_lr0.001/_sub_exp_20241115165948/retrain_bottom_model_A.pt", weights_only=False)
                retrain_bottom_model_B = torch.load("/home/user/bryanthx/exp/EE_resnet18_on_mri_lr0.001/_sub_exp_20241115165948/retrain_bottom_model_B.pt", weights_only=False)
                retrain_top_model = torch.load("/home/user/bryanthx/exp/EE_resnet18_on_mri_lr0.001/_sub_exp_20241115165948/retrain_top_model.pt", weights_only=False)

            elif args.data == "ct":
                full_bottom_model_A = torch.load("/scratch/taehongxi/LUV/exp/LUV_resnet18_on_ct_lr0.001/_sub_exp_20260207153225/full_bottom_model_A.pt", weights_only=False)
                full_bottom_model_B = torch.load("/scratch/taehongxi/LUV/exp/LUV_resnet18_on_ct_lr0.001/_sub_exp_20260207153225/full_bottom_model_B.pt", weights_only=False)
                full_top_model = torch.load("/scratch/taehongxi/LUV/exp/LUV_resnet18_on_ct_lr0.001/_sub_exp_20260207153225/full_top_model.pt", weights_only=False)
                retrain_bottom_model_A = torch.load("/home/user/bryanthx/exp/EE_resnet18_on_ct_lr0.001/_sub_exp_20241115165948/retrain_bottom_model_A.pt", weights_only=False)
                retrain_bottom_model_B = torch.load("/home/user/bryanthx/exp/EE_resnet18_on_ct_lr0.001/_sub_exp_20241115165948/retrain_bottom_model_B.pt", weights_only=False)
                retrain_top_model = torch.load("/home/user/bryanthx/exp/EE_resnet18_on_ct_lr0.001/_sub_exp_20241115165948/retrain_top_model.pt", weights_only=False)

            elif args.data == "custom":
                full_bottom_model_A = torch.load(f"{args.model_path}/full_bottom_model_A.pt", weights_only=False)
                full_bottom_model_B = torch.load(f"{args.model_path}/full_bottom_model_B.pt", weights_only=False)
                full_top_model = torch.load(f"{args.model_path}/full_top_model.pt", weights_only=False)
                retrain_bottom_model_A = torch.load(f"{args.model_path}/retrain_bottom_model_A.pt", weights_only=False)
                retrain_bottom_model_B = torch.load(f"{args.model_path}/retrain_bottom_model_B.pt", weights_only=False)
                retrain_top_model = torch.load(f"{args.model_path}/retrain_top_model.pt", weights_only=False)

            else:
                raise ValueError(f'No dataset named {args.data}!')
        
    elif args.model_type=='mixtext':
        full_bottom_model_A = torch.load("/scratch/taehongxi/LUV/exp/LUV_mixtext_on_yahoo_lr0.001/_sub_exp_20260207163427/full_bottom_model_A.pt", weights_only=False)
        full_bottom_model_B = torch.load("/scratch/taehongxi/LUV/exp/LUV_mixtext_on_yahoo_lr0.001/_sub_exp_20260207163427/full_bottom_model_B.pt", weights_only=False)
        full_top_model = torch.load("/scratch/taehongxi/LUV/exp/LUV_mixtext_on_yahoo_lr0.001/_sub_exp_20260207163427/full_top_model.pt", weights_only=False)
        retrain_bottom_model_A = torch.load("/home/user/bryanthx/exp/EE_mixtext_on_yahoo_lr0.001/_sub_exp_20241120221505/retrain_bottom_model_A.pt", weights_only=False)
        retrain_bottom_model_B = torch.load("/home/user/bryanthx/exp/EE_mixtext_on_yahoo_lr0.001/_sub_exp_20241120221505/retrain_bottom_model_B.pt", weights_only=False)
        retrain_top_model = torch.load("/home/user/bryanthx/exp/EE_mixtext_on_yahoo_lr0.001/_sub_exp_20241120221505/retrain_top_model.pt", weights_only=False)

    else:
        if args.data=='cifar10':
            full_bottom_model_A = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar10_lr0.001/full_train/full_bottom_model_A.pt", weights_only=False)
            full_bottom_model_B = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar10_lr0.001/full_train/full_bottom_model_B.pt", weights_only=False)
            full_top_model = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar10_lr0.001/full_train/full_top_model.pt", weights_only=False)
            retrain_bottom_model_A = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar10_lr0.001/retrain3/retrain_bottom_model_A.pt", weights_only=False)
            retrain_bottom_model_B = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar10_lr0.001/retrain3/retrain_bottom_model_B.pt", weights_only=False)
            retrain_top_model = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar10_lr0.001/retrain3/retrain_top_model.pt", weights_only=False)
            
        elif args.data == 'cifar100':
            full_bottom_model_A = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar100_lr0.001/full_train2/full_bottom_model_A.pt", weights_only=False)
            full_bottom_model_B = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar100_lr0.001/full_train2/full_bottom_model_B.pt", weights_only=False)
            full_top_model = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar100_lr0.001/full_train2/full_top_model.pt", weights_only=False)
            retrain_bottom_model_A = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar100_lr0.001/retrain3/retrain_bottom_model_A.pt", weights_only=False)
            retrain_bottom_model_B = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar100_lr0.001/retrain3/retrain_bottom_model_B.pt", weights_only=False)
            retrain_top_model = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar100_lr0.001/retrain3/retrain_top_model.pt", weights_only=False)
        else:
            raise ValueError(f'No model named {args.model_type}!')
    
    
    full_bottom_model_A = full_bottom_model_A.to(args.device)
    if full_bottom_model_B != None:
        full_bottom_model_B = full_bottom_model_B.to(args.device)
        full_top_model = full_top_model.to(args.device)

    retrain_bottom_model_A = retrain_bottom_model_A.to(args.device)
    if retrain_bottom_model_B != None:
        retrain_bottom_model_B = retrain_bottom_model_B.to(args.device)
        retrain_top_model = retrain_top_model.to(args.device)

    if args.data == 'mri':
        traindir = os.path.join('./data/Brain_MRI', 'Training')
        testdir = os.path.join('./data/Brain_MRI', 'Testing')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        trainset = datasets.ImageFolder(
                    traindir,
                    transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]))
        testset = datasets.ImageFolder(testdir, transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
        
    elif args.data == "ct":
        traindir = os.path.join('./data/Chest CT_v2', 'train')
        testdir = os.path.join('./data/Chest CT_v2', 'test')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        trainset = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
        testset = datasets.ImageFolder(testdir, transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]))
        
    elif args.data == 'yahoo':
        file_path = './data/yahoo_answers_csv/'
        trainset, _, _, _, _ = read_data_text.get_data(file_path, 5000)
        _, _, _, testset, _ = read_data_text.get_data(file_path, 10)
    elif args.data == 'custom':
        dataset = get_dataset(args)
        trainset = dataset.trainset
        testset = dataset.testset        
    else:
        dataset = get_dataset(args)
        trainset = deepcopy(dataset.trainset)
        testset = deepcopy(dataset.testset)

    full_train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    full_test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_to_forget = [args.unlearn_class]
    idx_train_forget = []
    idx_train_retain = []
    idx_test_forget = []
    idx_test_retain = []


    if args.data == 'yahoo':
        for i in range(len(trainset)):
            if trainset.labels[i] in class_to_forget:
                idx_train_forget.append(i)
            else:
                idx_train_retain.append(i)
    
        for i in range(len(testset)):
            if testset.labels[i] in class_to_forget:
                idx_test_forget.append(i)
            else:
                idx_test_retain.append(i)

    else:
        for i in range(len(trainset)):
            if args.unlearn_bias == 1: 
                if trainset.__biased__(i):
                    idx_train_forget.append(i)
                else:
                    idx_train_retain.append(i)
            else:
                if trainset.targets[i] in class_to_forget:
                    idx_train_forget.append(i)
                else:
                    idx_train_retain.append(i)
    
        for i in range(len(testset)):
            if testset.targets[i] in class_to_forget:
                idx_test_forget.append(i)
            else:
                idx_test_retain.append(i)

    retain_train_set = Subset(trainset, idx_train_retain)
    forget_train_set = Subset(trainset,idx_train_forget)
    retain_test_set = Subset(testset, idx_test_retain)
    forget_test_set = Subset(testset, idx_test_forget)

    if (args.unlearn_bias == 1):
        print(f'Unlearn Train Samples: {len(forget_train_set)}')

    train_forget_loader = DataLoader(forget_train_set, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_retain_loader = DataLoader(retain_train_set, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_forget_loader = DataLoader(forget_test_set, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_retain_loader = DataLoader(retain_test_set, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)


    if args.data == "cifar100":
        class_num = 100

    elif args.data in ["mri", "ct"]:
        class_num = 4

    else:
        class_num = 10

    print("class num : ", class_num)
    idx_subset_descent = [0] * class_num
    idx_limit = [3] * class_num
    idx_plus = []

    if args.data == 'yahoo':
        for i in range(len(trainset)):
            if trainset.labels[i] in class_to_forget:
                pass
            else:
                if idx_subset_descent[trainset.labels[i]] == idx_limit[trainset.labels[i]]:
                    pass
                else:
                    idx_subset_descent[trainset.labels[i]] = idx_subset_descent[trainset.labels[i]] + 1
                    idx_plus.append(i)

    else:
        for i in range(len(trainset)):
            #print("trainset targets : ", trainset.targets[i])
            if trainset.targets[i] in class_to_forget:
                #print("pass")
                pass
            else:
                #print("index descent b4 : ", idx_subset_descent)
                if idx_subset_descent[trainset.targets[i]] == idx_limit[trainset.targets[i]]:
                    #print("pass idx")
                    pass
                else:
                    idx_subset_descent[trainset.targets[i]] = idx_subset_descent[trainset.targets[i]] + 1
                    idx_plus.append(i)

    idx_few = []
    idx_subset = 0
    for i in range(len(forget_train_set)):
        if idx_subset == args.unlearn_samples:
            break
        else:
            idx_few.append(i)
            idx_subset = idx_subset + 1
    
    print("Samples used in LUV unlearn : ", idx_subset)

    fewshot_train_descent = Subset(trainset, idx_plus)
    fewshot_train = Subset(forget_train_set, idx_few)
    fewshot_descent_loader = DataLoader(fewshot_train_descent, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers) #batch size=129 for cifar100
    LUV_trainloader = DataLoader(fewshot_train, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    print("Finish Data Preprocessing")

    if (args.quick == False):

        acc_full = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, full_test_loader, args)
        acc_remain = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, test_retain_loader, args)
        acc_forget = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, test_forget_loader, args)

        acc_remain_retrain = evaluate(retrain_bottom_model_A, retrain_bottom_model_B, retrain_top_model, test_retain_loader, args)
        acc_forget_retrain = evaluate(retrain_bottom_model_A, retrain_bottom_model_B, retrain_top_model, test_forget_loader, args)

        print(f'Base Unbiased accuracy: {acc_remain}')
        run.log({'Base Unbiased acc' : acc_remain})
        print(f'Base Biased accuracy: {acc_forget}')
        run.log({'Base Biased acc' : acc_forget})

        print(f'Retrain Unbiased accuracy: {acc_remain_retrain}')
        run.log({'Retrain Unbiased acc' : acc_remain_retrain})
        print(f'Retrain Biased accuracy: {acc_forget_retrain}')
        run.log({'Retrain Biased acc' : acc_forget_retrain})

        metric = MIA(
            retain_loader=train_retain_loader,
            forget_loader=train_forget_loader,
            test_loader=full_test_loader,
            bottom_model_A=full_bottom_model_A,
            bottom_model_B = full_bottom_model_B,
            top_model=full_top_model,
            args=args
        )
        run.log({'Base MIA': metric * 100})

        metric = MIA(
            retain_loader=train_retain_loader,
            forget_loader=train_forget_loader,
            test_loader=full_test_loader,
            bottom_model_A=retrain_bottom_model_A,
            bottom_model_B = retrain_bottom_model_B,
            top_model=retrain_top_model,
            args=args
        )
        run.log({'Retrain MIA': metric * 100})

    if (args.unlearn_method == 'fisher'):
        unlearn_method = 'fisher'
    else:
        unlearn_method = unlearn.get_unlearn_method(args.unlearn_method)

    criterion = nn.CrossEntropyLoss()

    #Start unlearning
    start_time = time.time()
    if unlearn_method == 'fisher' :
        full_bottom_model_A, full_bottom_model_B = fisher_forgetting(full_bottom_model_A, full_bottom_model_B, full_top_model, train_retain_loader, train_forget_loader, test_retain_loader, 
                                                                     test_forget_loader, args, run, args.u_batch_size, DEVICE=args.device, evals=args.evals, Method=args.f_method)
    else :
        unlearn_method(LUV_trainloader, fewshot_descent_loader, full_bottom_model_A, full_bottom_model_B, full_top_model, criterion, class_to_forget, args, run, test_retain_loader, test_forget_loader, args.alpha)
    end_time = time.time()


    acc_remain = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, test_retain_loader, args)
    acc_biased = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, test_forget_loader, args)
    acc_forget = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, train_forget_loader, args)

    run.log({
        "Final Unbiased Class Accuracy" : acc_remain,
        "Final Biased Class Accuracy" : acc_biased,
        "Final Forgotten Label Accuracy" : acc_forget
        })

    duration = end_time - start_time
    duration = round(duration,2)
    run.log({"Training Time (s)" : duration})

    print(f"unbiased accuracy: {acc_remain}")
    print(f"biased accuracy: {acc_biased}")
    print(f"forgotten label accuracy: {acc_forget}")


    metric = MIA(
        retain_loader=train_retain_loader,
        forget_loader=train_forget_loader,
        test_loader=full_test_loader,
        bottom_model_A=full_bottom_model_A,
        bottom_model_B = full_bottom_model_B,
        top_model=full_top_model,
        args=args
    )
    run.log({'Unlearned MIA': metric * 100})



if __name__ == "__main__":
    main()