import numpy as np
# import torch
import cv2
# import png
import matplotlib . pyplot as plt
import time




def patch_image(image, patch_size):
    """_summary_
        divide the image into patchs;
    Args:
        image ( images ): original image with rows , columns and channels
        patch_size (_type_): the shape of the patch, ex patch_size * patch_size
    Returns:
        numpy array: pattch list 
    """
    
    patch_rgb = []
    patch_list = []
    patch_m_size = patch_size // 2
    
    # pad the whole image, respect to the edge, ex left most will connect to right most;
    image_pad = cv2.copyMakeBorder(image, patch_m_size, patch_m_size, patch_m_size, patch_m_size, cv2.BORDER_REFLECT)

    assert(len(image.shape) == 3)
    
    #divide the images into patches
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            for c in range(image.shape[2]):
                img_center = np.zeros(shape=(patch_size, patch_size))
                pixel_val = image[x][y][c]
                img_center.fill(pixel_val)
                
                x_real = x + patch_m_size
                y_real = y + patch_m_size

                x_i = x_real - patch_m_size
                x_s = x_real + patch_m_size + 1
                y_i = y_real - patch_m_size
                y_s = y_real + patch_m_size + 1

                curr_patch = image_pad[x_i:x_s, y_i:y_s, c] - img_center

                assert curr_patch.shape == (patch_size, patch_size)
                patch_rgb.append(curr_patch)
            patch_list.append(np.asarray(patch_rgb))
            patch_rgb = []
            
    np_patch_list = np.asarray(patch_list)

    return np_patch_list

def flow_patch_list_generator( patch_sizes_list, images):
    """for each patch size convert flow images to patchlized image 

    Args:
        patch_sizes_list (list): different patch sizes
        images (_type_): _description_

    Returns:
        list: patchlized image 
    """
    flow_patch_list = []
    
    for i in range(len(patch_sizes_list)):
        flow_patch_list.append(patch_image(images,patch_sizes_list[i]))
        
    return flow_patch_list

def evaluation_entry(fgim, gtim):
    """_summary_
    Args:
        fgim (_type_): _description_
        gtim (_type_): _description_
    Returns:
        _type_: _description_
    """

    if (len(fgim.shape) == 3):
        print("error: fgim mush be a gray image, fgim.shape:", fgim.shape)
        return -1, -1, -1, -1

    if (fgim.shape[0]*fgim.shape[1] != np.sum(fgim == 0) + np.sum(fgim == 255)):
        print("error: fgim is not clean")
        return -1, -1, -1, -1


    TP = np.sum((fgim == 255) & (gtim == 255))
    FP = np.sum((fgim == 255) & (gtim == 0))
    TN = np.sum((fgim == 0) & (gtim == 0))
    FN = np.sum((fgim == 0) & (gtim == 255))


    return TP, FP, TN, FN



def np_randomize_patch(patch):
    """ randomize shuffle on all pixels indexs  
        result good 
    Args:
        patch (image): 

    Returns:
        np array: randomized patch
    """
    # patch shape = (channel, size, size)
    
    channel, patch_height, patch_width = patch.shape
    # random_idx = np.random.randint(0, patch_width*patch_height, patch_width * patch_height)
    random_idx = np.arange(patch_width * patch_height)
    np.random.shuffle(random_idx)

    random_patch = np.zeros(shape=(patch_height * patch_width, channel))
    
    for c in range(channel):
     
        random_patch_flatten = np.ndarray.flatten(patch[c])
        #random_patch_c1 = np.zeros(shape=(patch_height * patch_width, ))
        
        random_patch_c = np.take(random_patch_flatten, random_idx)
        
        
        # for i in range(len(random_idx)):
        #     random_patch_c1[i] = random_patch_flatten[random_idx[i]]
        
        # if (np.array_equal(random_patch_c,random_patch_c1) ):
        #     print("i am equal")
        # else:
        #     print("rc",random_patch_c1)
        #     print ("rc",random_patch_c)
        
        random_patch[..., c] = random_patch_c
    random_patch_reshape = np.reshape(random_patch, newshape=(patch_height
                                                              ,patch_width, channel)).transpose(2, 0, 1)
    return random_patch_reshape


def randomize_patch_list(select_patch):
    """_summary_
        randomize the  patch list
    Args:
        select_patch (_type_): patch list, with patch shape (c, row, col)
    
    Returns:
        numpy: np_patch_list
    """
    random_patch_list = []
    for patch in select_patch:
        random_patch = np_randomize_patch(patch)
        #print(random_patch.shape)
        random_patch_list.append(random_patch)

    np_random_patch = np.asarray(random_patch_list).transpose(0, 2, 3, 1)
    return np_random_patch


def select_batch_size_patch(np_random_patch,patch_size,channel,batch_size, select_pixels_size):
    """_summary_
        #select batch size patches to train
    Args:
        np_random_patch (_type_): _description_
        patch_size (_type_): _description_
        channel (_type_): _description_
        batch_size (_type_): _description_
        pixel_size (_type_): _description_
    Returns:
        np: select_pixels
    """
    
    # flatten the patch. shape = (batch_size, patch_size*patch_size*channel)
    select_patch_flatten = np.reshape(np_random_patch, newshape=(batch_size, patch_size * patch_size, channel))
    
    select_pixels = np.zeros(shape=(batch_size, select_pixels_size * select_pixels_size, 3))
    R = select_patch_flatten[:, 0: select_pixels_size * select_pixels_size, 0]
    G = select_patch_flatten[:, 0: select_pixels_size * select_pixels_size, 1]
    B = select_patch_flatten[:, 0: select_pixels_size * select_pixels_size, 2]
    select_pixels[..., 0] = R
    select_pixels[..., 1] = G
    select_pixels[..., 2] = B
    
    return select_pixels

def image_resize(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    return resized


def network_inputGenerate(val, batch_size,channel, select_pixels_size,flow_patch_lists, patch_sizes_list):
    """_summary_
        generate each pixel's patch
        
    Args:
        val (_type_): _description_
        flow_patch_lists (list): contain each patchlized flow
        patch_size_lists  (list) : different patch sizes
    """
    
    #pixel's flow_patch_list 
    select_patch = flow_patch_lists[0][val * batch_size: (val + 1) * batch_size]
    #pixel's flow_patch_list_large
    select_patch_large = flow_patch_lists[1][val * batch_size: (val + 1) * batch_size]

    # randomize the patch
    np_random_patch = randomize_patch_list(select_patch)
    np_random_patch_large = randomize_patch_list(select_patch_large)
    


    # select batch size patches to train
    select_pixels = select_batch_size_patch(np_random_patch, patch_sizes_list[0], channel, batch_size,
                                            select_pixels_size)
    select_pixels_large = select_batch_size_patch(np_random_patch_large, patch_sizes_list[1], channel, batch_size,
                                                select_pixels_size)

    # select first L pixels
    # shape of batch_size, channel, select_pixels_size, select_pixels_size
    select_pixels_patch = np.reshape(select_pixels,
                                    newshape=(
                                    batch_size, select_pixels_size, select_pixels_size, channel)).transpose(0, 3, 1, 2)

    select_pixels_large_patch = np.reshape(select_pixels_large,
                                        newshape=(batch_size, select_pixels_size, select_pixels_size,
                                                    channel)).transpose(0, 3, 1,
                                                                2)

    np_random_select_pixel_list = randomize_patch_list(select_pixels_patch)
    np_random_select_pixel_list_large = randomize_patch_list(select_pixels_large_patch)

    # stack two list in channels dim, (1000,15,15,6)
    np_random_select_pixel = np.concatenate((np_random_select_pixel_list, np_random_select_pixel_list_large), axis=3)

    # reshape
    np_random_select_pixel = np_random_select_pixel.transpose(0, 3, 1, 2)
    
    return np_random_select_pixel
    
def show_img(img, flattened = False, ori_shape = (370, 1200)):

    if not flattened:
        print("img shape", img.shape)
        try:
            plt.imshow( img) 
            plt.pause(0.01)

        except Exception as exc:
            print("exc", exc)
    else:
        try:
            print("flatten img shape", img.shape, "old shape", ori_shape)
            np.reshape(img, (ori_shape))
            print("after de-flatten shape", img.shape)
            plt.imshow( img) 
            plt.pause(0.01)
        except Exception as exc:
            print("exc", exc)

def plot_graph(epoch_list, loss_list, fig_name):
    plt.plot(epoch_list, loss_list)
    plt.title("The Learning Curve of the Training Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(fig_name)
    plt.clf()


if __name__ == '__main__':
    # img = cv2.imread("/Users/david/Desktop/cmput414/project/pixel-distribution-learning/2011_09_26_drive_0001_sync_0000000000.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.random.rand(480, 854, 3)*255
    img = img.astype(np.uint8)

    # random_patch = np_randomize_patch(img)
    # #print(arr)
    # print(img.shape)

    
    patch = patch_image(img, 37)
    print(patch.shape)