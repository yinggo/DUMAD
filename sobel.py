import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import cv2

def edge_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 卷积输出通道，这里我设置为3
    # sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    # 输入图的通道，这里我设置为3
    # sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op(im)
    print(torch.max(edge_detect))
    # 将输出转换为图片格式
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

def edge_extraction():
    a = torch.Tensor([1, 1, 0])
    b = torch.Tensor([3, 2, 1])
    c = torch.sigmoid(b)
    print(c)
    loss = nn.BCELoss()
    print(loss(c,a))


    im = cv2.imread('/home/user/data4/chongxin/LIR-for-Unsupervised-IR/dataset/moni_motion/Celeba_A/HUANG_BIN-IMG-0002-00009.png', cv2.IMREAD_GRAYSCALE)
    im = np.expand_dims(im,axis = 0)
    # 添加一个维度，对应于pytorch模型张量(B, N, W, H)中的batch_size
    im = im[np.newaxis, :]
    im = torch.Tensor(im)
    edge_detect = edge_conv2d(im)
    out = np.zeros_like(edge_detect)
    cv2.normalize(edge_detect,out,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    out = np.uint8(out)
    cv2.imshow('edge.jpg', out)
    cv2.waitKey(0)
    # cv2.imwrite('edge-2.jpg', edge_detect)

if __name__ == "__main__":
    edge_extraction()
