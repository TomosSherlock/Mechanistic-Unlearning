import logging

logger = logging.getLogger(f"VFU.{__name__}")
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms


from dataset import get_dataset

from model.model_builder import get_model
from model import read_data_text

from utils import AvgMeter

from tqdm import tqdm

from copy import deepcopy
import math
from pathlib import Path




class MODEL(object):
    def __init__(self, args):
        self.args = args
        self.setup()
        self.test_acc_history = []
        self.best_test_acc = 0.0
        self.best_model = None


    def setup(self):
        
        if self.args.data.lower()=='yahoo':
            file_path = './data/yahoo_answers_csv/'
            train_set, _, _, _, _ = read_data_text.get_data(file_path, 5000)
            _, _, _, test_set, _ = read_data_text.get_data(file_path, 10)

        elif self.args.data.lower()=="mri":
            traindir = os.path.join('./data/Brain_MRI', 'Training')
            testdir = os.path.join('./data/Brain_MRI', 'Testing')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_set = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
            test_set = datasets.ImageFolder(testdir, transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]))
            
        elif self.args.data.lower()=="ct":
            traindir = os.path.join('./data/Chest CT_v2', 'train')
            testdir = os.path.join('./data/Chest CT_v2', 'test')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_set = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
            test_set = datasets.ImageFolder(testdir, transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]))

        elif self.args.data.lower()=="custom":
            dataset = get_dataset(self.args)
            train_set = dataset.trainset
            test_set = dataset.testset

        elif self.args.data.lower()=="custom-1":
            dataset = get_dataset(self.args)
            train_set = dataset.trainset
            test_set = dataset.testset            

        else:
            dataset = get_dataset(self.args)
            train_set = deepcopy(dataset.trainset)
            test_set = deepcopy(dataset.testset)
        
        # train a retrain model with remaining data
        if self.args.mode == 'retrain':
            idx_train_forget = []
            idx_train_retain = []
            idx_test_forget = []
            idx_test_retain = []
            if self.args.data == 'yahoo':
                class_to_forget = [self.args.unlearn_class]
                for i in range(len(train_set)):
                    if train_set.labels[i] in class_to_forget:
                        idx_train_forget.append(i)
                    else:
                        idx_train_retain.append(i)
    
                for i in range(len(test_set)):
                    if test_set.labels[i] in class_to_forget:
                        idx_test_forget.append(i)
                    else:
                        idx_test_retain.append(i)
            else:
                if self.args.unlearn_class_num == 1:
                    class_to_forget = [self.args.unlearn_class]
                elif self.args.unlearn_class_num == 2:
                    class_to_forget = [0,2]
                else: #4 unlearning classes. 
                    class_to_forget = [0,2,5,7]
                
                for i in range(len(train_set)):
                    if self.args.unlearn_bias == 1: 
                        if train_set.__biased__(i):
                            idx_train_forget.append(i)
                        else:
                            idx_train_retain.append(i)
                    else:
                        if train_set.targets[i] in class_to_forget:
                            idx_train_forget.append(i)
                        else:
                            idx_train_retain.append(i)

                for i in range(len(test_set)):
                    if test_set.targets[i] in class_to_forget:
                        idx_test_forget.append(i)
                    else:
                        idx_test_retain.append(i)

            train_set = Subset(train_set, idx_train_retain)
            test_set = Subset(test_set, idx_test_retain)

        self.train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.num_workers)
        self.test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=self.args.num_workers)
        

        self.model = get_model(args=self.args)[0]
        self.model = self.model.to(self.args.device)
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.train_acc_meter = AvgMeter()
        self.train_loss_meter = AvgMeter()
        self.test_acc_meter = AvgMeter()
        self.pc_train_acc_meter = [AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter()]
        self.pc_test_acc_meter = [AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter()]
        self.model_optimizer = torch.optim.SGD(self.model.parameters(),
                                                        self.args.optimizer_lr, 
                                                        momentum=self.args.momentum,
                                                        weight_decay=self.args.weight_decay)
        

    def train(self):
        self.model.train()
        self.train_acc_meter.reset()
        self.train_loss_meter.reset()

        # reset per-class accuracy
        for i in range(10):
            self.pc_train_acc_meter[i].reset()
        
        for step, (batch_x, batch_y) in tqdm(enumerate(self.train_loader)):
            if self.args.data != 'yahoo':
                batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
                x = batch_x
            elif self.args.data == 'custom-1':
                x = torch.stack([batch_x])
                x = torch.permute(x, (1,0,2,3))
            else:
                x = batch_x

            self.model_optimizer.zero_grad()
            output = self.model(x)

            loss = self.criterion(output, batch_y)
            
            acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
            self.train_acc_meter.update(acc.item(), batch_y.size(0))

            # update per-class accuracy
            for i in range(10):
                c_count = torch.sum(batch_y.cpu() == i)
                if (c_count > 0):
                    c_acc = torch.sum(torch.where(batch_y.cpu() == i, torch.argmax(output.cpu(), dim=-1) == batch_y.cpu(), torch.zeros(batch_y.size(0)))) / c_count 
                    self.pc_train_acc_meter[i].update(c_acc, c_count)

            loss.backward()
            self.model_optimizer.step()
            self.train_loss_meter.update(loss.item(), output.size(0))




    def test(self):
        self.model.eval()
        self.test_acc_meter.reset()

        # reset per-class accuracy
        for i in range(10):
            self.pc_test_acc_meter[i].reset()

        labels = []
        preds = []
        with torch.no_grad():
            for step, (batch_x, batch_y) in tqdm(enumerate(self.test_loader)):

                if self.args.data != 'yahoo':
                    batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
                    x = batch_x
                elif self.args.data == 'custom-1':
                        x = torch.stack([batch_x])
                        x = torch.permute(x, (1,0,2,3))
                else:
                    x = batch_x
            
                output = self.model(x)

                acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
                self.test_acc_meter.update(acc.item(), batch_y.size(0))
                self.test_acc_history.append(self.test_acc_meter.get())

                # update per-class accuracy
                for i in range(10):
                    c_count = torch.sum(batch_y.cpu() == i)
                    if (c_count > 0):
                        c_acc = torch.sum(torch.where(batch_y.cpu() == i, torch.argmax(output.cpu(), dim=-1) == batch_y.cpu(), torch.zeros(batch_y.size(0)))) / c_count 
                        self.pc_test_acc_meter[i].update(c_acc, c_count)

            if self.test_acc_meter.get() > self.best_test_acc:
                self.best_test_acc = self.test_acc_meter.get()
                self.best_model = deepcopy(self.model)


    def adjust_lr(self):
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] *= self.args.gamma

    def save(self):
        save_path = f'{self.args.outdir}/{self.args.epochs}'
        out_dir = Path(save_path)
        if (out_dir.is_dir() == False) :
            os.mkdir(out_dir)
        print(f'Models Saved at {out_dir}')

        if self.args.mode == 'retrain':
            model_path = os.path.join(save_path, 'retrain_standard_model.pt')
            torch.save(self.best_model.cpu(), model_path)
        else:
            model_path = os.path.join(save_path, 'full_standard_model.pt')
            torch.save(self.best_model.cpu(), model_path)