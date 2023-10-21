import numpy as np
from skimage.util import random_noise
import cv2



def load_gray_img(filepath):
    img = cv2.imread(
        filepath, cv2.IMREAD_GRAYSCALE
        )
    # print(img.shape)
    img = np.expand_dims(
        img, axis=2
    )
    # print(img.shape)
    return img

def add_noise_skimage(img, sigma):
    img = np.float64(img)
    noisy_img = random_noise(
        img, mode='gaussian', 
        rng=0, 
        clip=False, 
        mean=0, var=sigma**2
        )
    # img, [0, 255], float
    return noisy_img


def add_noise(img, sigma):

    # make sure noise added to all images are the same
    np.random.seed(0) 

    gaussian_noise = np.random.normal(0, sigma, img.shape)
    noisy_img = img + gaussian_noise
    return noisy_img