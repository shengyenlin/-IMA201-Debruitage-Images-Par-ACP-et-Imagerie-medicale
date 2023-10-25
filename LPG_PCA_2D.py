# set arg parser -> store hyperparameter
# set PSNR, SSIM evaluation -> automation

import argparse
import os
from pathlib import Path
import time
from itertools import product

import logging

import numpy as np
import cv2
import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from sklearn.feature_extraction import image

from metrics import calculate_psnr, calculate_ssim, skim_compare_psnr, skim_compare_ssim
from utils import add_noise, load_gray_img, add_noise_skimage

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_name", type=str, default="experiment_log.txt")

    parser.add_argument("--input_dir", type=Path, default="./input/clean")
    parser.add_argument("--sigmas", type=int, nargs="+", help = "noise level of images")
    parser.add_argument("--output_dir", type=Path, default="./output")

    # hyperparameter
    parser.add_argument("--Ks", type=int, nargs="+", default=3)
    parser.add_argument("--Ls", type=int, nargs="+", default=9)
    parser.add_argument("--cs", type= int, nargs="+", default=8)
    parser.add_argument("--c_s", help="estimation error of noiseless images", default=0.35, nargs="+", type=float)

    parser.add_argument("--store_image", action='store_true')

    args = parser.parse_args()
    return args


def vector_pixel(x,y,k,image):
    # generate a vector variable of a pixel with coordinate (x,y) 
    # and window length k
    halfk = k // 2
    l = []
    for i in range(y-halfk, y+halfk+1):
        for j in range(x-halfk, x+halfk+1):
            l.append(image[j][i])
    return np.array(l)

def LPG_error(array1, array2):
    # calculate the difference 
    # between vector variable 
    # and training sample
    err = 0
    for i in range(len(array1)):
        err += (array1[i] - array2[i])**2
    return err/len(array1)

def get_block_for_one_pixel(img, x, y, half_k):
    block = img[x-half_k: x+half_k+1, y-half_k: y+half_k+1]
    return block

def get_all_training_features(img, x, y, K, L):
    dim1, dim2 = img.shape
    half_l = L // 2

    # deal with edges
    x_min = 0 if x-half_l < 0 else x-half_l
    x_max = dim1 if x+half_l > dim1 else x+half_l
    y_min = 0 if y-half_l < 0 else y-half_l
    y_max = dim2 if y+half_l > dim2 else y+half_l


    training_block = img[
        x_min:x_max, y_min:y_max
        ]
    training_features= image.extract_patches_2d(
        training_block, (K, K)
    ).reshape(-1, K, K)

    return training_features

def get_PCA_training_features(c, K, training_features, target):

    # Sort by MSE
    cm = c * (K ** 2)
    n = cm if cm < training_features.shape[0] else training_features.shape[0]

    square_err = ((training_features - target)**2)
    mse = np.mean(
        square_err.reshape(-1, K**2), axis=1
    )

    sort_indexes = np.argsort(mse)

    # (n, K^2)
    training_features_PCA = training_features[sort_indexes[:n], :, :] \
        .reshape(n, target.shape[0]**2)
    return training_features_PCA

def PCA_denoise(X, sigma):
    
    X = X.swapaxes(1, 0) # (K^2, n)
    X_mean = np.mean(X, axis=1).reshape(-1, 1)   # (K^2, )
    X = X - X_mean

    cov_sigma = sigma**2 * np.eye(X.shape[0], X.shape[0]) # sigma^2 * I, (K^2, K^2)
    sigma_X = np.cov(X) # sigma_x^bar, (K^2, K^2)
    eigen_X = np.linalg.eig(sigma_X)[1] # phi_x_bar, (K^2, K^2)
    PX = eigen_X.T # 3.9 (K^2, K^2)

    Y_v_bar = PX @ X # 3.9 - 2, (K^2, n)
    sigma_v = PX @ cov_sigma @ PX.T # 3.7 - 2, (K^2, K^2)

    # correspond to "In implementation, we first calculate ..."
    sigma_y_v_bar = (Y_v_bar @ Y_v_bar.T)/X.shape[0] # 3.10
    phi_y_bar = np.maximum( 
        np.zeros(sigma_y_v_bar.shape), 
        sigma_y_v_bar - sigma_v 
        ) # 3.12
    
    # dim = (K^2, )
    shrinkage_coef = np.diag(
        phi_y_bar
        )/(np.diag(phi_y_bar) + np.diag(sigma_y_v_bar)) # 3.12
    # )/(np.diag(phi_y_bar) + np.diag(sigma_v)) # 3.12
    
    # dim = (K^2, )
    denoise_X = PX.T @ (Y_v_bar * shrinkage_coef.reshape(-1, 1)) # 3.13
    denoise_X += X_mean
    denoise_pixel = denoise_X[denoise_X.shape[0]//2, 0] # retrieves the element in the middle of the X1 array
    return denoise_pixel

def denoise_one_pixel(img, x, y, K, L, c, sigma):
    # x, y = position of denoised pixel
    half_k = K // 2
    half_l = L // 2

    # Block centered around x,y, dim = (K, K)
    target_block = get_block_for_one_pixel(img, x, y, half_k)
    
    # All Training features, dim = (-1, K, K)
    all_training_features = get_all_training_features(img, x, y, K, L)
    # print(all_training_features.shape)

    # sort and select top n, dim = (n, K^2)
    PCA_features = get_PCA_training_features(c, K, all_training_features, target_block)

    # denoise, dim = (K^2, )
    denoise_pixel = PCA_denoise(PCA_features, sigma)
    return denoise_pixel

def denoise_image(img, K, L, c, sigma): 
    half_k = K//2
    out_img = np.copy(img)
    for x in range(half_k, img.shape[0] - half_k):
        for y in range(half_k, img.shape[1] - half_k):
            out_img[x, y] = denoise_one_pixel(img, x, y, K, L, c, sigma)
    
    return out_img

def denoise_image_gray_scale_two_stage(img, k, l, c, c_s, sigma):

    stage_1_denoised_img = denoise_image(img, k, l, c, sigma)

    # should be sigma ** 2?
    sigma_2 = c_s * np.sqrt(sigma - np.mean(
        (img - stage_1_denoised_img)**2)
    )

    stage_2_denoised_img = denoise_image(stage_1_denoised_img, k, l, c, sigma_2)
    return stage_1_denoised_img, stage_2_denoised_img

def denoise_image_2D(img, k, l, c, c_s, sigma):
    if img.ndim == 2:    # grey image
        return denoise_image_gray_scale_two_stage (img, k, l, c, c_s, sigma)
    else:    # 3 channels image
        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]

        stage_1_denoised_img_r, stage_2_denoised_img_r = \
            denoise_image_gray_scale_two_stage(img_r, k, l, c, c_s, sigma)
        stage_1_denoised_img_g, stage_2_denoised_img_g = \
            denoise_image_gray_scale_two_stage(img_g, k, l, c, c_s, sigma)
        stage_1_denoised_img_b, stage_2_denoised_img_b = \
            denoise_image_gray_scale_two_stage(img_b, k, l, c, c_s, sigma)

        stage_1_denoised_img = np.zeros(img.shape, dtype=np.uint8)
        stage_1_denoised_img[:,:,2] = stage_1_denoised_img_r
        stage_1_denoised_img[:,:,1] = stage_1_denoised_img_g
        stage_1_denoised_img[:,:,0] = stage_1_denoised_img_b

        stage_2_denoised_img = np.zeros(img.shape, dtype=np.uint8)
        stage_2_denoised_img[:,:,2] = stage_2_denoised_img_r
        stage_2_denoised_img[:,:,1] = stage_2_denoised_img_g
        stage_2_denoised_img[:,:,0] = stage_2_denoised_img_b  

        return stage_1_denoised_img, stage_2_denoised_img


def main():

    args = parse_args()

    # Configure the logging
    logging.basicConfig(
        level=logging.INFO, filename = args.log_name, 
        format='%(message)s', filemode='w')
    
    in_images_rel = [f for f in os.listdir(args.input_dir)]

    hyper_param_product = list(product(args.Ks, args.Ls, args.cs, args.c_s))

    for sigma in args.sigmas:
        logging.info(f"Start denoising with sigma = {sigma}")
        out_dir = os.path.join(args.output_dir, f"gauss_{sigma}")
        os.makedirs(out_dir, exist_ok=True)

        # normalize sigma
        sigma = sigma/255.0

        for K, L, c, c_s in hyper_param_product:
            logging.info(f"Hyerparameter: K = {K}, L = {L}, c = {c}, c_s = {c_s}")

            # normalize noise
            x = time.time() 
            for img_path in in_images_rel:

                # in original code, denoise on normalized image
                in_path = os.path.join(args.input_dir, img_path)
                clean_img = io.imread(in_path).astype('float64')/255.0
                noisy_img = add_noise(clean_img, sigma)

                stage_1_denoised_img, stage_2_denoised_img = denoise_image_2D(
                    noisy_img, K, L, c, c_s, sigma
                )         

                y = time.time()

                n_axis = clean_img.ndim - 1

                psnr_stage_1 = round(
                    skim_compare_psnr(clean_img, stage_1_denoised_img), 3
                    )
                psnr_stage_2 = round(
                    skim_compare_psnr(clean_img, stage_2_denoised_img), 3
                )
                ssim_stage_1 = round(
                    skim_compare_ssim(clean_img, stage_1_denoised_img, n_axis), 3
                )
                ssim_stage_2 = round(
                    skim_compare_ssim(clean_img, stage_2_denoised_img, n_axis), 3
                )

                logging.info(f"{img_path} | First stage - PNSR: {psnr_stage_1}, SSIM: {ssim_stage_1} | Second stage - PNSR: {psnr_stage_2}, SSIM: {ssim_stage_2} | {round((y-x)/60, 4)} mins used")
                if args.store_image:
                    out_path_noisy = os.path.join(out_dir, f"noisy_{img_path}")
                    out_path_1 = os.path.join(out_dir, f"stage_1_k_{K}_l_{L}_c_{c}_{img_path}")
                    out_path_2 = os.path.join(out_dir, f"stage_2_k_{K}_l_{L}_c_{c}_{img_path}")

                    cv2.imwrite(out_path_noisy, noisy_img)
                    cv2.imwrite(out_path_1, stage_1_denoised_img)      
                    cv2.imwrite(out_path_2, stage_2_denoised_img)

if __name__ == "__main__":
    main() 