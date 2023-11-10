# IMA201 Projet - PCA Image denoising

Python Implementation of LPG-PCA algorithm [1] and an extension to 3D images.

## Environment

Python=3.10.13 \
matplotlib=3.8.0 \
networkx=3.1 \
numpy=1.26.0 \
opencv-python=4.8.1.78 \
scikit-image=0.22.0 \
scikit-learn=1.3.1 \
scipy=1.11.3

## Usage

Set the value of hyperparameters in .bash file, where 
* K : the length of variable block (central block) which models a pixel and its nearest neighbors
* L : the length of training block in which training samples are selected according to the similarity to the central block
* c_s : estimation error of noiseless images
* sigma : noise level of images when using gaussian noise

To test the code on 2D image, run "bash lpg_pca_2D.bash";  To test the code on 3D image, run "bash lpg_pca_3D.bash".


[1] Lei Zhang, Weisheng Dong, David Zhang, and Guangming Shi. Two-stage image denoising by principal component analysis with local pixel grouping. Pattern Recognition, 43(4):1531-1549, 2010.
