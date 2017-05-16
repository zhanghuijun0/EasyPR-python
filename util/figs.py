import cv2
import os

def imwrite(dir_name, name, img):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(os.path.join(dir_name, name), img)

def imshow(name, img):
    cv2.imshow(name, img)
    0xFF & cv2.waitKey()