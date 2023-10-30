import numpy as np
from skimage.util import random_noise
import cv2

class MinMaxScaler3D:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None

    def fit(self, X):
        self.min_ = np.min(X)
        self.max_ = np.max(X)

        # Compute the scaling factor
        range_min, range_max = self.feature_range
        self.scale_ = (range_max - range_min) / (self.max_ - self.min_)

    def transform(self, X):
        # Apply the scaling transformation
        return (X - self.min_) * self.scale_ + self.feature_range[0]

    def inverse_transform(self, X):
        # Apply the inverse scaling transformation
        return (X - self.feature_range[0]) / self.scale_ + self.min_

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


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.float64)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def add_noise(img, sigma, type='gaussian'):

    # make sure noise added to all images are the same
    np.random.seed(0) 
    if type == 'gaussian':
        noise = np.random.normal(0, sigma, img.shape)
        noisy_img = img + noise
    elif type == 'sp':
        noisy_img = sp_noise(img, sigma)
    
    return noisy_img