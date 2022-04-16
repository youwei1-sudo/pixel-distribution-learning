import os
import time
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from dataloader import *
from utils import *
from net import *

# import matplotlib . pyplot as plt

# Please change your root here
#data_root = "../KITTI_MOD_fixed"
data_root = "/media/zlu6/4caa1062-1ae5-4a99-9354-0800d8a1121d/KITTI_MOD_fixed"
model_path = "./checkpoint/ckpt.pth"

imgs = load_flow_images(root=data_root, mode="training")
print(imgs.shape)

train_sets = imgs[183: 243]
nums_train = len(train_sets)
train_idx = np.arange(nums_train)
np.random.shuffle(train_idx)
train_sets_shuffled = np.take(train_sets, train_idx, axis=0)

validate_sets = imgs[183: 243]

#show_img(train_sets[0])
print(train_sets.shape)

masks = load_masks(root=data_root, mode="training")

train_masks = masks[183: 243]
train_masks_shuffled = np.take(train_masks, train_idx, axis=0)
validate_masks = masks[183: 243]

# show_img(validate_masks[0])
print(train_masks.shape)

_, row, column, channel = imgs.shape

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))

epochs = 50
batch_size = 2000
patch_size = 25
patch_size_larger = 37


patch_sizes_list = [patch_size, patch_size_larger]
# divisble by kernel size
select_pixels_size = 16

net = Net().to(device)

# TODO: test Adam or SGD
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
class_weights = torch.FloatTensor([0.3, 0.7]).to(device)
criterion = torch.nn.NLLLoss(class_weights, reduction="mean").to(device)

best_fscore = 0

validate_dir = os.path.join(os.getcwd(), "validate_img")
if not os.path.exists(validate_dir):
    os.makedirs(validate_dir)

if not os.path.exists("checkpoint"):
    os.makedirs("checkpoint")

for epoch in range(epochs):
    total_loss = 0
    for i in range(0, len(train_sets_shuffled), 2):
     
        flow_patch_lists = flow_patch_list_generator(patch_sizes_list, train_sets_shuffled[i])
    
        mask_image = train_masks_shuffled[i].reshape(row * column) / 255

        # generate patch for each pixel in an image
        value = round(flow_patch_lists[0].shape[0] / batch_size + 0.5)

        value_idx = np.arange(value)
        np.random.shuffle(value_idx)

        # calculate how many batches to iterate
        for val in value_idx:
        # for val in range(value):
            
            # pixel corresponding mask value
            mask_patch = mask_image[val * batch_size: (val + 1) * batch_size]
            
            # reshape
            np_random_select_pixel = network_inputGenerate(val, batch_size,channel, select_pixels_size,flow_patch_lists, patch_sizes_list)
            # print ("shape of np_random_select_pixel",np_random_select_pixel.shape)

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
    # TODO: set validate condition
    if epoch % 2 == 0:
        pred_list = []
        mask_list = []
        current_fscore = 0
        total_fscore = 0
        with torch.no_grad():
            for i in range(0, len(validate_sets), 4):

                flow_patch_list = patch_image(validate_sets[i], patch_size)
                flow_patch_list_large = patch_image(validate_sets[i], patch_size_larger)

                # flow_patch_list = patch_image(validate_sets[i], patch_size)
                mask_image = validate_masks[i]

                # generate patch for each pixel in an image
                value = round(flow_patch_list.shape[0] / batch_size + 0.5)
                # calculate how many batches to iterate

                for val in range(value):
                    select_patch = flow_patch_list[val * batch_size: (val + 1) * batch_size]
                    select_patch_large = flow_patch_list_large[val * batch_size: (val + 1) * batch_size]

                    # pixel corresponding mask value
                    mask_patch = mask_image[val * batch_size: (val + 1) * batch_size]

                    # randomize the patch
                    np_random_patch = randomize_patch_list(select_patch)
                    np_random_patch_large = randomize_patch_list(select_patch_large)
                    # np_random_patch = np.asarray(random_patch_list).transpose(0, 2, 3, 1)
                    # np_random_patch_large = np.asarray(random_patch_large_list).transpose(0, 2, 3, 1)

                    # select batch size patches to train
                    select_pixels = select_batch_size_patch(np_random_patch, patch_size, channel, batch_size,
                                                            select_pixels_size)
                    select_pixels_large = select_batch_size_patch(np_random_patch_large, patch_size_larger, channel,
                                                                  batch_size, select_pixels_size)

                    # select first L pixels
                    # shape of batch_size, channel, select_pixels_size, select_pixels_size
                    select_pixels_patch = np.reshape(select_pixels,
                                                     newshape=(batch_size, select_pixels_size, select_pixels_size,
                                                               channel)).transpose(0, 3, 1,
                                                                                   2)
                    select_pixels_large_patch = np.reshape(select_pixels_large,
                                                           newshape=(batch_size, select_pixels_size, select_pixels_size,
                                                                     channel)).transpose(0, 3, 1,
                                                                                         2)

                    np_random_select_pixel_list = randomize_patch_list(select_pixels_patch)
                    np_random_select_pixel_list_large = randomize_patch_list(select_pixels_large_patch)

                    # np_random_select_pixel_list = np.asarray(random_select_pixel_list)
                    # np_random_select_pixel_list_large = np.asarray(random_select_pixel_list_large)

                    # stack two list in channels dim, (1000,15,15,6)
                    np_random_select_pixel = np.concatenate((np_random_select_pixel_list, np_random_select_pixel_list_large),
                                                            axis=3)

                    # reshape
                    np_random_select_pixel = np_random_select_pixel.transpose(0, 3, 1, 2)

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

                Re = TP / (TP + FN+ 0.001)
                Pr = TP / (TP + FP+ 0.001)
                Fm = (2 * Pr * Re) / (Pr + Re + 0.001)
                total_fscore += Fm
                print("validate img index", i, "Re:", Re, " Pr:", Pr, " Fm:", Fm)

        current_fscore = total_fscore / (len(validate_sets) // 4)
        print("epoch:", epoch, "avg Fm: ", current_fscore, "best fscore:", best_fscore)
        if best_fscore < current_fscore:
            best_fscore = current_fscore
            torch.save(net.to(device).state_dict(), model_path)
            print("save the model in epoch %d" % epoch)
