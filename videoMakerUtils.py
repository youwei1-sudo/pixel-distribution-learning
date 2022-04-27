# importing cv2
import cv2


def my_put_text(img, fm):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    image2 = cv2.putText(img, "fm%.2f" % fm, org, font,
                         fontScale, color, thickness, cv2.LINE_AA)
    return image2