import numpy as np
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


def add_noise(img, sigma):
    original_img = np.float64(img) 
    original_img_ = original_img/255.

    # make sure noise added to all images are the same
    np.random.seed(0) 

    # if img.shape[0] == 3:
    #     H, W, C = original_img.shape[0], original_img.shape[1], original_img.shape[2]
    #     N = H * W * C
    # else:
    #     H, W = original_img.shape[0], original_img.shape[1]
    #     N = H * W

    gaussian_noise = np.random.normal(0, sigma/255., original_img.shape)
    noisy_img = original_img_ + gaussian_noise

    noisy_img_un_normalized = (noisy_img.clip(0, 1) * 255.0).round()
    return noisy_img_un_normalized