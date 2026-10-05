import logging
import os

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from PIL import Image
logger = logging.getLogger("VFU." + __name__)


def get_dataset(args):
    if args.data.lower() == "cifar10":
        dataset = CIFAR10(args)
    elif args.data.lower() == "cifar100":
        dataset = CIFAR100(args)
    elif args.data.lower() == "mnist":
        dataset = MNIST(args)
    else:
        raise ValueError(f'No dataset named {args.data}!')
    
    return dataset



class Dataset(object):
    def __init__(self, args):
        self.args = args




class CIFAR10(Dataset):
    def __init__(self, args):
        super(CIFAR10, self).__init__(args)
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        self.train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                                   transforms.ToTensor(),
                                                   normalize,
                                                   ])
        self.test_transform = transforms.Compose([transforms.CenterCrop(32),
                                                  transforms.ToTensor(),
                                                  normalize,
                                                  ])
        self.trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True,
                                         transform=self.train_transform)
        self.testset = datasets.CIFAR10(root=args.data_path, train=False, download=True,
                                        transform=self.test_transform)



class MNIST(Dataset):
    def __init__(self, args):
        super(MNIST, self).__init__(args)
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.trainset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=self.train_transform)
        self.testset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=self.test_transform)


class CIFAR100(Dataset):
    def __init__(self, args):
        super(CIFAR100, self).__init__(args)
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        self.train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                                   transforms.ToTensor(),
                                                   normalize,
                                                   ])
        self.test_transform = transforms.Compose([transforms.CenterCrop(32),
                                                  transforms.ToTensor(),
                                                  normalize,
                                                  ])
        self.trainset = datasets.CIFAR100(root=args.data_path, train=True, download=True,
                                         transform=self.train_transform)
        self.testset = datasets.CIFAR100(root=args.data_path, train=False, download=True,
                                        transform=self.test_transform)