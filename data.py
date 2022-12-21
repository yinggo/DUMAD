"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
from skimage import io as IO
import torch
import matplotlib.pyplot as plt
def default_loader(path):

    imag = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    tem = np.zeros(imag.shape, dtype=np.float32)

    cv2.normalize(imag, tem, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    tem = np.expand_dims(tem, axis=0)

    tem = torch.from_numpy(tem)

    return tem
    # return Image.open(path)
    # return Image.open(path).convert('RGB')
    IO.imread()
    

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


# class ImageLabelFilelist(data.Dataset):
#     def __init__(self, root, flist, transform=None,
#                  flist_reader=default_flist_reader, loader=default_loader):
#         self.root = root
#         self.imlist = flist_reader(os.path.join(self.root, flist))
#         self.transform = transform
#         self.loader = loader
#         self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
#         self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
#         self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]
#
#     def __getitem__(self, index):
#         impath, label = self.imgs[index]
#         img = self.loader(os.path.join(self.root, impath))
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data
import cv2
from PIL import Image
import os
from torch.autograd import Variable
import os.path
import numpy as np
import random
import glob
import scipy.io as sio
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            # ima = torch.unsqueeze(img,0)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

class myImageFolder(data.Dataset):

    def __init__(self, root):
        mat = sio.loadmat(root)
        name_tem = root.split('/')[-1]
        name = name_tem.split('.')[0]
        data = mat[name]
        data = np.float32(data)
        out = np.zeros(data.shape, dtype=np.float32)
        for i in range(data.shape[0]):
            data1 = data[i,:,:]
            tem1 = np.zeros(data1.shape, dtype=np.float32)
            cv2.normalize(data1, tem1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            out[i, :, :] = tem1[:,:]
        self.root = root
        self.imgs = out

    def __getitem__(self, index):
        img = self.imgs[index]
        height = 128
        width = 128
        tem_height = random.randint(10,img.shape[0]-height-10)
        tem_width = random.randint(10,img.shape[1]-width-10)
        patch = img[tem_height:tem_height+128,tem_width:tem_width+128]

        patch = np.expand_dims(patch, axis=0)
        # input = Variable(torch.from_numpy(patch).type(Tensor),dtype = torch.float64)
        input = torch.from_numpy(patch)
        return input

    def __len__(self):
        return len(self.imgs)


class mymotionImageFolder(data.Dataset):

    def __init__(self, root):
        seqs_dirs = sorted(glob.glob(os.path.join(root, '*')))
        sequences_motion = []
        for seq_dir in seqs_dirs:
            mat = sio.loadmat(seq_dir)
            # name_tem = 'motion'
            name_tem = seq_dir.split('/')[-2]
            data=np.float32(mat[name_tem])
            # data = np.float32(mat['motion'])
            # data = data[:, 1, :, :]

            data = np.float32(data)
            for i in range(0,data.shape[0]):
                tem = data[i,:,:]
                tem1 = np.zeros(tem.shape, dtype=np.float32)
                cv2.normalize(tem, tem1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                sequences_motion.append(tem1)

        self.motion = sequences_motion


    def __getitem__(self, index):
        motion = self.motion[index]
        height = 128
        width = 128
        tem_height = random.randint(10,motion.shape[0]-height-10)
        tem_width = random.randint(10,motion.shape[1]-width-10)
        patch = motion[tem_height:tem_height+128,tem_width:tem_width+128]

        patch = np.expand_dims(patch, axis=0)
        # input = Variable(torch.from_numpy(patch).type(Tensor),dtype = torch.float64)
        input = torch.from_numpy(patch)
        return input

    def __len__(self):
        return len(self.motion)

class mymotionImageFolder2(data.Dataset):

    def __init__(self, root):
        seqs_dirs = sorted(glob.glob(os.path.join(root, '*')))
        sequences_motion = []
        for seq_dir in seqs_dirs:
            mat = sio.loadmat(seq_dir)
            # name_tem = 'motion'
            name_tem = seq_dir.split('/')[-2]
            data = np.float32(mat[name_tem])
            # data = np.float32(mat['gt'])
            # data = data[:, :, :, 1]

            data = np.float32(data)
            for i in range(0,data.shape[0]):
                tem = data[i,:,:]
                tem1 = np.zeros(tem.shape, dtype=np.float32)
                cv2.normalize(tem, tem1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                sequences_motion.append(tem1)

        self.motion = sequences_motion


    def __getitem__(self, index):
        motion = self.motion[index]
        height = 128
        width = 128
        tem_height = random.randint(10,motion.shape[0]-height-10)
        tem_width = random.randint(10,motion.shape[1]-width-10)
        patch = motion[tem_height:tem_height+128,tem_width:tem_width+128]

        patch = np.expand_dims(patch, axis=0)
        # input = Variable(torch.from_numpy(patch).type(Tensor),dtype = torch.float64)
        input = torch.from_numpy(patch)
        return input

    def __len__(self):
        return len(self.motion)