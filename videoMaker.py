import cv2
import os
import re

image_folder = '/Users/david/Documents/cmput414/project/pixel-distribution-learning/test_output_imgs_0416'
video_name = 'video0418.avi'

images = []
# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

for img in os.listdir(image_folder) :
    print(img)
    dot_idx =img.find(".")
    img_name =  img[:dot_idx]

    if img.endswith(".png")  and (int(img_name)%2 != 0):
       
        images.append(img)

images = sorted(images,key=lambda x: int(os.path.splitext(x)[0]))
print(images)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()