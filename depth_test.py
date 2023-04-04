import cv2
import numpy as np

image_RGB = cv2.imread(r'Z:\assistant\dataset\input_image_raw_trim\20230106_105004.png', cv2.IMREAD_COLOR)
image_depth = cv2.imread(r'Z:\assistant\dataset\input_Image_depth_trial1\904429898944667648_1.png', cv2.IMREAD_ANYDEPTH)
image_depth_gray = cv2.cvtColor(cv2.convertScaleAbs(image_depth, alpha=(255.0 / 65535.0)), cv2.COLOR_GRAY2BGR)

concatenate_img = np.hstack((image_RGB, image_depth_gray))

cv2.imshow('RGB_depth', concatenate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

TEST = 1

