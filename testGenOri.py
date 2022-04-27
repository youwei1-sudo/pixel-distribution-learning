import numpy as np
import cv2
from utils import *
from dataloader import *
from net import *
from videoMakerUtils import *

data_root = "../KITTI_MOD_fixed"
#data_root = "/media/zlu6/4caa1062-1ae5-4a99-9354-0800d8a1121d/KITTI_MOD_fixed/"


imgs = load_flow_images(root=data_root, mode="training")
imgs = imgs[182: 296]

masks = load_masks(root=data_root, mode="training")
masks = masks[182: 296]
batch_size = 2000
patch_size = 25
patch_size_larger = 37
select_pixels_size = 16

_, row, column, channel = imgs.shape

test_dir = os.path.join(os.getcwd(), "test_output_imgs_0416")
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

mask_dir = os.path.join(os.getcwd(), "mask_imgs_0416")
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def testGen():

    # model.eval()

    video_name = 'GtMaskvideo0418.avi'
    frame = masks[0]
    height, width = frame.shape
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    
    with torch.no_grad():
        for i in range(1, len(masks), 2):
            if (i > 55):
                break
            video.write(masks[i])
            
    cv2.destroyAllWindows()
    video.release()
if __name__ == '__main__':
    testGen()