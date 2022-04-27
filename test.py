import numpy as np
import cv2
from utils import *
from dataloader import *
from net import *
from videoMakerUtils import my_put_text
import time
import argparse
from videoMakerUtils import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", help="directory of training dataset")
parser.add_argument("--model_path", help="directory of saved model")
parser.add_argument("--test_image_dir", help="directory to save test image result")
parser.add_argument("--gt_dir", help="directory to save ground truth images")
args = parser.parse_args()


data_root = args.data_root


imgs = load_flow_images(root=data_root, mode="training")
imgs = imgs[182: 296]

masks = load_masks(root=data_root, mode="training")
masks = masks[182: 296]
batch_size = 2000
patch_size = 25
patch_size_larger = 37
select_pixels_size = 16

_, row, column, channel = imgs.shape

test_dir = args.test_image_dir
# test_dir = os.path.join(os.getcwd(), "test_output_imgs_0424")
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# mask_dir = os.path.join(os.getcwd(), "mask_imgs_0424")
mask_dir = args.gt_dir
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def test():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
    model = Net().to(device)

    #checkpoint = torch.load('./checkpoint_0416/ckpt.pth')
    checkpoint = torch.load(args.model_path, map_location=device)

    
    #CPU test
    # model = torch.load('./checkpoint_0417/ckpt_0.pth')
    
    model.load_state_dict(checkpoint)
    # model.eval()
    pred_list = []
    total_fscore = 0

    with torch.no_grad():
        for i in range(1, len(imgs), 2):
            start_time = time.time()
            flow_patch_list = patch_image(imgs[i], patch_size)
            flow_patch_list_large = patch_image(imgs[i], patch_size_larger)

            # flow_patch_list = patch_image(validate_sets[i], patch_size)
            mask_image = masks[i]
            cv2.imwrite(os.path.join(mask_dir, "mask_%d.png" % i), mask_image)

            # generate patch for each pixel in an image
            value = round(flow_patch_list.shape[0] / batch_size + 0.5)
            # calculate how many batches to iterate

            for val in range(value):
                # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                np_random_select_pixel = np.concatenate(
                    (np_random_select_pixel_list, np_random_select_pixel_list_large),
                    axis=3)

                # reshape
                np_random_select_pixel = np_random_select_pixel.transpose(0, 3, 1, 2)

                # reshape selected pixels
                # np.random.shuffle(select_pixels_patch)
                # randomize again to avoid overfitting
                model.eval()
                input_patch = torch.FloatTensor(np_random_select_pixel)
                input_patch = input_patch.to(device)
                output = model(input_patch)
                batch_pred_labels = torch.argmax(output, axis=1)
                batch_pred_labels = batch_pred_labels.cpu().numpy()
                pred_list += list(batch_pred_labels)

            pred_image = np.asarray(pred_list)
            prefgim = pred_image.reshape(row, column).astype(np.uint8) * 255

            print("--- %s seconds ---" % (time.time() - start_time))
            TP, FP, TN, FN = evaluation_entry(prefgim, mask_image)
            pred_list = []

            Re = TP / (TP + FN + 0.001)
            Pr = TP / (TP + FP + 0.001)
            Fm = (2 * Pr * Re) / (Pr + Re + 0.001)

            # cv2.putText(prefgim, "Fm%.2f" % Fm, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            my_put_text(prefgim, Fm)
            cv2.imwrite(os.path.join(test_dir, "%d.png" % i), prefgim)

            print("validate img index", i, "Re:", Re, " Pr:", Pr, " Fm:", Fm)


if __name__ == '__main__':
    test()