import os
import cv2
import numpy as np
import torch

import utils
from utils import *
from davis_dataparser import Davis_DataParser
from tqdm import tqdm


def load_flow_images(root, resolution, category):
    dataset_object = Davis_DataParser(data_root=root, resolution=resolution, category=category)
    data_path = dataset_object.data_path_list
    num_frame = len(data_path)
    im = cv2.imread(data_path[0].flow_image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im_cropped = im[0: 400, 0:800]
    im_scaled = utils.image_resize(im_cropped, 50)

    assert(len(im.shape) == 3)
    row, col, channel = im_scaled.shape

    imgs = np.empty([num_frame, row, col, channel], dtype=im_scaled.dtype)
    # num_frame = 1
    for i in tqdm(range(num_frame)):
        # print(data_path[i].flow_image)
        img = cv2.imread(data_path[i].flow_image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_cropped = img_rgb[0:400, 0:800]
        im_scaled = utils.image_resize(img_cropped, 50)
        imgs[i] = im_scaled
    return imgs


def load_masks(root, resolution, category):
    dataset_object = Davis_DataParser(data_root=root, resolution=resolution, category=category)
    data_path = dataset_object.data_path_list
    num_frame = len(data_path)
    im = cv2.imread(data_path[0].semantic_label)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_cropped = im[0:400, 0:800]
    im_scaled = utils.image_resize(im_cropped, 50)

    assert(len(im_scaled.shape) == 2)
    row, col = im_scaled.shape
    channel = 1

    imgs = np.empty([num_frame, row, col, channel], dtype=im_scaled.dtype).squeeze()

    # num_frame = 1
    for i in tqdm(range(num_frame)):
        img = cv2.imread(data_path[i].semantic_label)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_cropped = img_gray[0:400, 0:800]
        im_scaled = utils.image_resize(img_cropped, 50)
        imgs[i] = im_scaled
    return imgs


if __name__ == '__main__':
    data_root = "/media/zlu6/4caa1062-1ae5-4a99-9354-0800d8a1121d/DAVIS-data/DAVIS"
    resolution = "480p"
    category = "kitesurf-flows"
    imgs = load_flow_images(root=data_root, resolution=resolution, category=category)
    print(imgs.shape)

    masks = load_masks(root=data_root, resolution=resolution, category=category                                                                                                           )
    print(masks.shape)


