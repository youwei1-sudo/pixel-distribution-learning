import os
import time
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import utils
import cv2
from dataloader import *
from utils import *
from net import *

# TODO:
# 1. save the model according to the best F-score
# 2. implement the test function. If choose to test the model only, run test() without training
# 3. test the model on another dataset
# 4. stack the patch with a larger patch
# 5. tune hyperparams, patch size and network
# 6. code formatting

# Optional
# 1. stack the optical flow's patch with origin image's patch
# 2. rewrite the patch_image() function (by tiling?), this version is too slow



data_root = "/media/zlu6/4caa1062-1ae5-4a99-9354-0800d8a1121d/KITTI_MOD_fixed"
model_path = "./checkpoint/ckpt.pth"
imgs = load_flow_images(root=data_root, mode="training")
train_sets = imgs[190: 195]
validate_sets = imgs[200: 202]
print(train_sets.shape)

masks = load_masks(root=data_root, mode="training")
train_masks = masks[190: 195]
validate_masks = masks[200: 202]
print(train_masks.shape)

_, row, column, channel = imgs.shape

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
epochs = 10
batch_size = 1000
patch_size = 25
patch_size_large = 45
pixel_size = 15
net = Net().to(device)
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# class_weights = torch.FloatTensor([0.5, 0.5]).to(device)
# criterion = torch.nn.NLLLoss(weight=class_weights, reduction='mean').to(device)

criterion = torch.nn.NLLLoss().to(device)
validate_dir = os.path.join(os.getcwd(), "validate_img")
if not os.path.exists(validate_dir):
    os.makedirs(validate_dir)

if not os.path.exists("checkpoint"):
    os.makedirs("checkpoint")

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(train_sets)):
    # for i in range(1):
        flow_patch_list = patch_image(train_sets[i], patch_size)
        # flow_patch_list_large = patch_image(train_sets[i], patch_size_large)
        mask_image = train_masks[i].reshape(row * column) / 255
        # generate patch for each pixel in an image
        value = round(flow_patch_list.shape[0] / batch_size + 0.5)
        # calculate how many batches to iterate
        for val in range(value):
            random_patch_list = []
            random_select_pixel_list = []
            # random_large_patch_list = []
            # random_select_large_pixel_list = []

            select_patch = flow_patch_list[val * batch_size: (val+1) * batch_size]
            # select_patch_large = flow_patch_list_large[val * batch_size: (val+1) * batch_size]
            mask_patch = mask_image[val * batch_size: (val+1) * batch_size]
            # gt_patch = mask_patch_list[val * batch_size: (val+1) * batch_size]
            # select batch size patches to train
            for patch in select_patch:
                random_patch = utils.randomize_patch(patch)
                random_patch_list.append(random_patch)
            np_random_patch = np.asarray(random_patch_list).transpose(0, 2, 3, 1)

            # np.random.shuffle(select_patch)
            # randomize the patch
            select_patch_flatten = np.reshape(np_random_patch, newshape=(batch_size, patch_size * patch_size, channel))
            # flatten the patch. shape = (batch_size, patch_size*patch_size*channel)
            select_pixels = np.zeros(shape=(batch_size, pixel_size*pixel_size, 3))
            R = select_patch_flatten[:, 0: pixel_size*pixel_size, 0]
            G = select_patch_flatten[:, 0: pixel_size*pixel_size, 1]
            B = select_patch_flatten[:, 0: pixel_size*pixel_size, 2]
            select_pixels[..., 0] = R
            select_pixels[..., 1] = G
            select_pixels[..., 2] = B
            # select_pixels = select_patch_flatten[:, 0: pixel_size*pixel_size*channel]
            # select first L pixels
            select_pixels_patch = np.reshape(select_pixels, newshape=(batch_size, pixel_size, pixel_size, channel)).transpose(0, 3, 1, 2)
            # reshape selected pixels
            # np.random.shuffle(select_pixels_patch)
            for pixel_patch in select_pixels_patch:
                random_pixel_patch = utils.randomize_patch(pixel_patch)
                random_select_pixel_list.append(random_pixel_patch)
            np_random_select_pixel = np.asarray(random_select_pixel_list)

            # randomize again to avoid overfitting
            net.train()
            input_patch = torch.FloatTensor(np_random_select_pixel)
            input_patch = input_patch.to(device)
            target = torch.tensor(mask_patch, dtype=torch.int64)
            target = target.to(device)

            output = net(input_patch)

            loss = criterion(output, target)

            total_loss = total_loss + loss.item()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print("finish train image %d" % i)
            # print("value:", val, "patch loss:", loss)
    print("epoch:", epoch, " loss:", total_loss)
    if epoch % 1 == 0:
        pred_list = []
        mask_list = []
        current_fscore = 0
        best_fscore = 0
        total_fscore = 0
        with torch.no_grad():
            for i in range(len(validate_sets)):
                flow_patch_list = patch_image(validate_sets[i], patch_size)
                mask_image = validate_masks[i]
                # mask_image = validate_masks[i].reshape(row * column) / 255
                # mask_list.append(mask_image)
                # generate patch for each pixel in an image
                value = round(flow_patch_list.shape[0] / batch_size + 0.5)
                # calculate how many batches to iterate
                for val in range(value):
                    random_patch_list = []
                    random_select_pixel_list = []
                    select_patch = flow_patch_list[val * batch_size: (val + 1) * batch_size]
                    # mask_patch = mask_image[val * batch_size: (val + 1) * batch_size]
                    # gt_patch = mask_patch_list[val * batch_size: (val+1) * batch_size]
                    # select batch size patches to train
                    # np.random.shuffle(select_patch)
                    # randomize the patch
                    for patch in select_patch:
                        random_patch = utils.randomize_patch(patch)
                        random_patch_list.append(random_patch)
                    np_random_patch = np.asarray(random_patch_list).transpose(0, 2, 3, 1)
                    select_patch_flatten = np.reshape(np_random_patch,
                                                      newshape=(batch_size, patch_size * patch_size, channel))
                    # flatten the patch. shape = (batch_size, patch_size*patch_size*channel)
                    select_pixels = np.zeros(shape=(batch_size, pixel_size * pixel_size, 3))
                    R = select_patch_flatten[:, 0: pixel_size * pixel_size, 0]
                    G = select_patch_flatten[:, 0: pixel_size * pixel_size, 1]
                    B = select_patch_flatten[:, 0: pixel_size * pixel_size, 2]
                    select_pixels[..., 0] = R
                    select_pixels[..., 1] = G
                    select_pixels[..., 2] = B
                    # select_pixels = select_patch_flatten[:, 0: pixel_size*pixel_size*channel]
                    # select first L pixels
                    select_pixels_patch = np.reshape(select_pixels,
                                                     newshape=(batch_size, pixel_size, pixel_size, channel)).transpose(0, 3, 1, 2)
                    for pixel_patch in select_pixels_patch:
                        random_pixel_patch = utils.randomize_patch(pixel_patch)
                        random_select_pixel_list.append(random_pixel_patch)
                    np_random_select_pixel = np.asarray(random_select_pixel_list)
                    # reshape selected pixels
                    # np.random.shuffle(select_pixels_patch)
                    # randomize again to avoid overfitting
                    net.eval()
                    input_patch = torch.FloatTensor(np_random_select_pixel)
                    input_patch = input_patch.to(device)
                    output = net(input_patch)
                    batch_pred_labels = torch.argmax(output, axis=1)
                    batch_pred_labels = batch_pred_labels.cpu().numpy()
                    pred_list += list(batch_pred_labels)
                pred_image = np.asarray(pred_list)
                prefgim = pred_image.reshape(row, column).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(validate_dir, "epoch%d_%d.png") % (epoch, i), prefgim)
                TP, FP, TN, FN = evaluation_entry(prefgim, mask_image)
                pred_list = []

                Re = TP / (TP + FN)
                Pr = TP / (TP + FP)
                Fm = (2 * Pr * Re) / (Pr + Re)
                total_fscore += Fm

                print("validate img index", i, "Re:", Re, " Pr:", Pr, " Fm:", Fm)
        current_fscore = total_fscore / len(validate_sets)
        print("epoch:", epoch, "avg Fm: ", current_fscore)
        if best_fscore < current_fscore:
            best_fscore = current_fscore
            torch.save(net.state_dict(), model_path)





