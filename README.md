# pixel-distribution-learning
Motion Segmentation is the task of identifying the independently moving objects (pixels) in the video and separating them from the background motion.

Our work include use the state of the art : Pixel distribution model (random feature selection) to preprocess the extracted optical flow images. Then we will serve the extracted pixel distributions as inputs to our motion segmentation net. Finally, we will generate our own segementation binary masks against groud truths.

Tested dataset:
http://www.cvlibs.net/datasets/kitti/

![alt text](https://github.com/youwei1-sudo/pixel-distribution-learning/blob/main/documents/git_docs/workFlow.jpg)
