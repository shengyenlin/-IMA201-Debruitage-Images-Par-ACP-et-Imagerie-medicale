# set arg parser -> store hyperparameter
# set PSNR, SSIM evaluation -> automation

import argparse
import os
from pathlib import Path
import time

import numpy as np
import cv2
import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from sklearn.feature_extraction import image

from metrics import calculate_psnr, calculate_ssim
from utils import add_noise, load_gray_img

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=Path, default="./input/clean")
    parser.add_argument("--sigma", type=int, help = "noise level of images")
    parser.add_argument("--output_dir", type=Path, default="./output")

    # hyperparameter
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--L", type=int, default=9)
    parser.add_argument("--T", help = "threshold", type=int, default=20)
    parser.add_argument("--c", type= int, default=8)
    # parser.add_argument("--n", help = "number of training samples", type=int, default=3)
    parser.add_argument("--c_s", help="estimation error of noiseless images", default=0.35, type=float)

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

def select_training_samples(x,y,k,l,t,image,cm):
    # select training samples for the central pixel (x,y)
    # with the window size k and l, threshold t
    l_training_samples = []
    l_error = []
    halfk = k // 2
    halfl = l // 2

    # TODO: don't need to compute the center block every time
    for i in range(y-halfl+halfk, y+halfl-halfk+1):    #y
        for j in range(x-halfl+halfk, x+halfl-halfk+1):    #x
            error = LPG_error(
                vector_pixel(j,i,k,image), 
                vector_pixel(x,y,k,image)
                )
            if j!= x or i != y:
                l_training_samples.append(vector_pixel(j,i,k,image))
                l_error.append(error)
    
    pairs = zip(l_training_samples, l_error)
    
    # Sort the pairs based on the values in l_error
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    
    # Unpack the sorted pairs to get the sorted l_training_samples
    sorted_samples = [pair[0] for pair in sorted_pairs]
    sorted_error = [pair[1] for pair in sorted_pairs]
    
    #find the largest error in error list that are smaller than T
    index_error = 0
    while sorted_error[index_error] < t and index_error < len(sorted_error)-1:
        index_error += 1
            
    # make sure we have at least cm training examples
    if index_error < cm:
        sorted_samples = sorted_samples[:cm]
    else:
        sorted_samples = sorted_samples[:index_error]
    
    return np.array(sorted_samples)

def pca_denoise(array_original, array_samples, n_pc):
#     if not array_samples.any():
#         # Handle the case where array_samples is empty
#         return array_original

#     if n_pc > min(array_samples.shape[0], array_samples.shape[1]):
#         n_pc = min(array_samples.shape[0], array_samples.shape[1])

    pca = PCA(n_components=n_pc)
    pca.fit(array_samples)
    vector_pixel_pca = pca.transform(array_original.reshape(1,-1))
    vector_pixel_denoised = pca.inverse_transform(vector_pixel_pca)
    return vector_pixel_denoised


def denoise_picture(image, k, l, t, n_pc, cm):
    # denoise image, return new values of every pixel
    halfk = k//2
    halfl = l//2
    l_denoised_value = []
    length_edge = halfl - halfk + 1
    for i in range(image.shape[0]):    #y
        for j in range(image.shape[0]):    #x
            # we only denoise pixels that are not in the edge of image
            if i < length_edge or i >= image.shape[0]-length_edge \
                  or j < length_edge or j >= image.shape[0]-length_edge:
                l_denoised_value.append(image[j][i])
            else:
                l_denoised_value.append(
                    pca_denoise(
                        vector_pixel(j,i,k,image), 
                        select_training_samples(j,i,k,l,t,image,cm), n_pc)[0][k**2//2]
                        )
    return l_denoised_value

def denoise_first_ver():
    # l_denoised_value = denoise_picture(
    #     im, args.K, args.L, 
    #     args.T, args.N_PC, args.CM
    #     )
    
    # ll = 256
    # denoised_im = []
    # for i in range(ll):
    #     denoised_im.append([])
    #     for j in range(ll):
    #         denoised_im[i].append(l_denoised_value[i*ll + j])
    # denoised_im = np.array(denoised_im)

    # # TODO: fixed
    # denoised_im = np.rot90(denoised_im,k=3)
    # denoised_im = np.flip(denoised_im, axis=1)
    pass

def get_block_for_one_pixel(img, x, y, half_k):
    # TODO: we don't need to deal with edges here?
    block = img[x-half_k: x+half_k+1, y-half_k: y+half_k+1]
    return block

def get_all_training_features(img, x, y, K, L):
    dim1, dim2 = img.shape
    half_l = L // 2

    # deal with edgesf
    x_min = 0 if x-half_l < 0 else x-half_l
    x_max = dim1 if x+half_l > dim1 else x+half_l
    y_min = 0 if y-half_l < 0 else y-half_l
    y_max = dim2 if y+half_l > dim2 else y+half_l

    training_block = img[x_min:x_max, y_min:y_max]
    training_features= image.extract_patches_2d(
        training_block, (K, K)
    ).reshape(-1, K, K)
    return training_features

def get_PCA_training_features(c, K, training_features, target):
    # Sort by MSE
    n = c * (K ** 2)
    square_diff = np.mean(
        (training_features - target)**2, axis=(1,2)
    )
    sort_indexes = np.argsort(square_diff)

    # (n, K^2)
    training_features_PCA = training_features[sort_indexes[:n]] \
        .reshape(n, target.shape[0]**2)
    return training_features_PCA

def PCA_denoise(X, sigma):
    
    X = X.swapaxes(1, 0) # (K^2, n)
    X_mean = np.mean(X, axis=1)   # (K^2, )
    X = X - X_mean.reshape(-1, 1)

    cov_sigma = sigma**2 * np.eye(X.shape[0], X.shape[0]) # sigma^2 * I, (K^2, K^2)
    sigma_X = np.cov(X) # sigma_x^bar, (K^2, K^2)
    # sigma_X = (X.T @ X)/X.shape[0] 
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
        )/(np.diag(phi_y_bar) + np.diag(sigma_v)) # 3.12
    
    # dim = (K^2, )
    # print(shrinkage_coef.shape, Y_v_bar.shape, PX.shape)
    denoise_X = PX.T @ (Y_v_bar[:, 0] * shrinkage_coef) # 3.13
    # print(denoise_X.shape, X_mean.shape)
    denoise_X += X_mean
    # print(denoise_X.shape)
    # print(denoise_X)
    denoise_pixel = denoise_X[denoise_X.shape[0]//2] # retrieves the element in the middle of the X1 array
    return denoise_pixel

def denoise_one_pixel(img, x, y, K, L, c, sigma):
    half_k = K // 2
    half_l = L // 2

    # Block centered around x,y, dim = (K, K)
    target_block = get_block_for_one_pixel(img, x, y, half_k)
    
    # All Training features, dim = (-1, K, K)
    all_training_features = get_all_training_features(img, x, y, K, L)

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

def main():
    args = parse_args()

    in_images_rel = [f for f in os.listdir(args.input_dir)]
    # in_images_abs = [os.path.join(input_dir, f) for f in in_images_rel]

    out_dir = os.path.join(args.output_dir, f"gauss_{args.sigma}")
    os.makedirs(out_dir, exist_ok=True)
    x = time.time() 
    for img_path in in_images_rel:
        print(f"Denoising {img_path}", end = " ,")
        img_path = os.path.join(args.input_dir, img_path)
        clean_img = io.imread(img_path)
        noisy_img = add_noise(clean_img, args.sigma)

        print(clean_img.shape)
        print(noisy_img.shape)

        stage_1_denoised_img = denoise_image(
            noisy_img, args.K, args.L, 
            args.c, args.sigma
            )
        
        ## TODO: problem?
        sigma_2 = args.c_s * np.sqrt(
            args.sigma**2 - np.mean(
                (noisy_img - stage_1_denoised_img)**2
                )
            )
        
        # print(sigma_2)
        
        stage_2_denoised_img = denoise_image(
            stage_1_denoised_img, args.K, args.L, 
            args.c, sigma_2
            )
        
        out_path_1 = os.path.join(out_dir, f"stage_1_{img_path}")
        out_path_2 = os.path.join(out_dir, f"stage_2_{img_path}")

        cv2.imwrite(out_path_1, stage_1_denoised_img)      
        cv2.imwrite(out_path_2, stage_2_denoised_img)
        
        y = time.time()
        print(f"{round((y-x)/60, 4)} mins used")
        # calculate PSNR, SSIM
        psnr_stage_1 = calculate_psnr(clean_img, stage_1_denoised_img)
        psnr_stage_2 = calculate_psnr(clean_img, stage_2_denoised_img)
        ssim_stage_1 = calculate_ssim(clean_img, stage_1_denoised_img)
        ssim_stage_2 = calculate_ssim(clean_img, stage_2_denoised_img)

        print(f"First stage denoise result - PNSR: {psnr_stage_1}, SSIM: {ssim_stage_1}")
        print(f"Second stage denoise result - PNSR: {psnr_stage_2}, SSIM: {ssim_stage_2}")

if __name__ == "__main__":
    main() 