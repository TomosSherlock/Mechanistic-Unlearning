import logging
import os

import numpy as np
import torch
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
    elif args.data.lower() == "custom":
        dataset = CUSTOM(args)
    elif args.data.lower() == "custom-1":
        dataset = CUSTOM(args)
    else:
        raise ValueError(f'No dataset named {args.data}!')
    
    return dataset



class Dataset(object):
    def __init__(self, args):
        self.args = args

def get_position(indices, shape):
    '''Given the indices for an element and
    the shape of the array, return the position
    in memory of the element, ranging from 0
    to the number of elements in the array minus 1

    :param indices: an array of indices [i1,...,in]
    :param shape: an array of axis dimensions [d1,...,dn]
    '''

    # start with last index
    pos = indices[-1] 

    # Handle all other indices
    for i in range(0, len(shape) - 1):

        # calculate dimension product
        prod = 1
        for j in range(i + 1, len(shape)):
            prod *= shape[j]

        pos += indices[i] * prod

    return pos


class IdxDataset(Dataset):
    '''Custom dataset class for converting from MNIST-like idx3/1 files to a dataset,
    parsing guide for idx files at joshuazingale.net

    :param filename: the path to the file
    '''
    def __init__(self, filename : str):
        self.f = open(filename, "rb")

        # check if the file is an idx
        m_num = self.f.read(2)
        if (m_num != b'\x00\x00'):
            raise OSError(f'{filename} is not in idx format; magic number bytes {m_num}')

        # check the datatype of the file
        data_type = self.f.read(1)
        if (data_type == b'\x08'):
            self.data_type = torch.uint8 
            data_size = 1
        else :
            raise OSError(f'cannot parse datatype of {filename}')

        # number of axes
        n_axes = int.from_bytes(self.f.read(1), byteorder = 'big')

        # see how each dimension is formatted
        shape = []
        for _ in range (n_axes):
            shape.append(int.from_bytes(self.f.read(4),byteorder='big'))

        self.shape = tuple(shape)
        # print(f'image data shape: {shape}')

        # Get the size of one entry in elements
        self.entry_size = 1
        for dim in self.shape[1:]:
            self.entry_size *= dim
        # Get size of one entry in bytes
        self.entry_size_b = self.entry_size * data_size

        # start point of data in file
        self.data_start = self.f.tell()

    

    def __len__(self):
        return self.shape[0]

    def __getitem__(self,index):
        address = self.data_start + index * self.entry_size_b
        self.f.seek(address)

        # Get an entry from memory
        out = torch.frombuffer(
            bytearray(self.f.read(self.entry_size)),
            dtype = self.data_type,
            count = self.entry_size
        )

        out = out.reshape(self.shape[1:])

        if (self.entry_size > 1) :
            # Rescale to be on [-1,1]
            out = (out - (out.max()/2)) / (out.max()/2)

        # Reshape data
        return out.reshape(self.shape[1:])


class CombinationDataset(torch.utils.data.Dataset):
    '''Dataset for combining two datasets,
    one being the inputs and one being the labels'''

    def __init__(self, dataset, labelset):
        self.data = dataset
        self.targets = labelset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class BiasedDataset(torch.utils.data.Dataset):
    '''Dataset for combining three datasets,
    one being the inputs, one being the labels, and one being whether or not each input is biased'''

    def __init__(self, dataset, labelset, biasedset):
        self.data = dataset
        self.targets = labelset
        self.biased = biasedset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __biased__(self, idx):
        return self.biased[idx]



# read an idx file as dataset, instead of built-in datasets
class CUSTOM(Dataset):
    def __init__(self, args):
        super(CUSTOM, self).__init__(args)
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_images = IdxDataset(args.data_path + "/" + args.train_prefix + args.data_suffix)
        train_labels = IdxDataset(args.data_path + "/" + args.train_prefix + args.label_suffix)
        biased = IdxDataset(args.data_path + "/train_biased.idx1-ubyte")
        self.trainset = BiasedDataset(train_images, train_labels, biased)

        test_images = IdxDataset(args.test_path + "/" + args.test_prefix + args.data_suffix)
        test_labels = IdxDataset(args.test_path + "/" + args.test_prefix + args.label_suffix)
        self.testset = CombinationDataset(test_images, test_labels)

        print(self.trainset.data.shape)
        print(self.testset.data.shape)

        # self.trainset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=self.train_transform)
        # self.testset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=self.test_transform)

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