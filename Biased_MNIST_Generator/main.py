import os
import torch


from configs import config
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import shutil 
import struct


def extend_channels(data):
    '''Convert w*h (black and white) images into w*h*3 images.s
    '''
    ex_data = data
    ex_data = torch.stack([ex_data,ex_data,ex_data], dim=1)
    return ex_data.permute(0,1,2,3)



def _binary_to_colour(data, labels):
    # colours to paint each biased class
    COLOUR_MAP = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]])

    colour = COLOUR_MAP[labels.long()]
    colour = colour.view(data.shape[0],1,1,3)

    fg_data = torch.zeros_like(data)
    fg_data[data != 0] = 255
    fg_data[data == 0] = 0
    fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)

    bg_data = torch.zeros_like(data)
    bg_data[data == 0] = 1
    bg_data[data != 0] = 0
    bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
    bg_data = bg_data * colour
    bg_data = bg_data.permute(0,3,1,2)

    # print(fg_data.shape, bg_data.shape)

    data = fg_data + bg_data
    return data
    # return data.permute(0, 2, 3, 1)

class IdxDataset():
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
        match data_type:
            case b'\x08':
                self.data_type = torch.uint8 

                data_size = 1
            case _:
                raise OSError(f'cannot parse datatype of {filename}')

        # number of axes
        n_axes = int.from_bytes(self.f.read(1), byteorder = 'big')

        # see how each dimension is formatted
        shape = []
        for _ in range (n_axes):
            shape.append(int.from_bytes(self.f.read(4),byteorder='big'))

        self.shape = tuple(shape)

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

        # Reshape data
        return out.reshape(self.shape[1:])

# save data as .idx file
def save_data(data, path):
    """
    Saves a PyTorch tensor as a standard IDX binary file.
    """
    data = data.cpu().to(torch.uint8)
    data_type_magic = 0x08 
    num_dimensions = data.ndim

    header_magic = bytes([0, 0, data_type_magic, num_dimensions])
    
    header_dimensions = b""
    for dim_size in data.shape:
        header_dimensions += struct.pack('>I', dim_size)
    raw_data_bytes = data.numpy().tobytes()
    
    with open(path, "wb") as f:
        f.write(header_magic)       
        f.write(header_dimensions)    
        f.write(raw_data_bytes)       


if __name__ == '__main__':
    # handle CLI arguments
    args = config()

    # colours to paint each biased class
    COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]

    # extract images and labels from provided idx files
    train_images = IdxDataset(args.images_path)
    train_labels = IdxDataset(args.labels_path)

    images = DataLoader(train_images, batch_size=len(train_images), shuffle=False)
    images = next(iter(images))

    labels = DataLoader(train_labels, batch_size=len(train_labels), shuffle=False)
    labels = next(iter(labels))

    # mask representing things to recolour
    mask = (labels < args.n_biased_classes).view(len(train_images),1,1,1)

    # apply bias with rho probability
    rho_tensor = torch.full_like(mask, args.rho, dtype=float)
    drop_mask = torch.bernoulli(rho_tensor).to(torch.bool)
    mask *= drop_mask

    # apply background colouring to samples to become biased
    biased_images = torch.where(mask, 
                         _binary_to_colour(images, labels), 
                         extend_channels(images))
    biased_images = biased_images.to(torch.uint8)
 
    # create directory to save samples
    sample_dir_path = Path(f'{args.sample_path}/{args.n_biased_classes}-{args.rho}')
    if (sample_dir_path.is_dir()) :
        shutil.rmtree(sample_dir_path)
    os.mkdir(f'{args.sample_path}/{args.n_biased_classes}-{args.rho}')

    # save preview images
    n_previews = min(biased_images.shape[0], args.samples)
    for i in range(n_previews):
        preview_image = to_pil_image(biased_images[i].cpu())
        preview_image.save(f'{args.sample_path}/{args.n_biased_classes}-{args.rho}/b_mnist-{i}-class_{labels[i]}-{mask[i][0][0][0]}.png')

    labels = labels.flatten()
    mask = mask.flatten()

    # create directory to save binary data files
    out_dir = Path(f'{args.data_output_path}/{args.n_biased_classes}-{args.rho}')
    if (out_dir.is_dir()) :
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    # save biased datapoints
    save_data(biased_images, f'{out_dir}/train-images.idx3-ubyte')
    save_data(labels, f'{out_dir}/train-labels.idx1-ubyte')
    save_data(mask, f'{out_dir}/train_biased.idx1-ubyte')

