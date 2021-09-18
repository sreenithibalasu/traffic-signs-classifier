import cv2
import numpy as np
#Converting image to grayscale
def check_consistency(x, y, type):
    assert(x.shape[0] == y.shape[0]), 'The number of images in {} set does not match the number of labels'.format(type)
    assert(x.shape[1:] == (32, 32, 3)), 'The dimensions of the images in {} set are not 32x32x3'.format(type)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocess(img):
    img = to_gray(img)
    img = equalize(img)
    img = img/255
    return img
