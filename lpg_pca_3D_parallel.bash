python3 LPG_PCA_3D_parallel.py \
    --sigmas 10 20 30 40 \
    --Ks 3 5 7 \
    --Ls 19 20 21 22 23\
    --cs 8 \
    --c_s 0.35 0.5 0.65 \
    --log_name log_1030_kidney_MRI.txt \
    --input_dir IMA201_Medical_images/Kidney_MRI_KITS23/case_00000/image_could_be_denoised \
    --output_dir output_1030_kidney\
    --store_image \
    --kidney_data;

