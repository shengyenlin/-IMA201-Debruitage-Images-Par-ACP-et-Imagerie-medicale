python3 LPG_PCA_2D.py \
    --noise_type gaussian \
    --sigmas 14.73 \
    --Ks 3 \
    --Ls 19 \
    --cs 8 \
    --c_s 0.35 0.5 0.65 \
    --log_name real_noise_ultrasound_2ch.txt \
    --input_dir IMA201_Medical_images/Cardiac_US_CAMUS/tmp_2ch \
    --store_image;


# python3 LPG_PCA_2D.py \
#     --noise_type sp \
#     --prob 0.05 \
#     --Ks 3 \
#     --Ls 19 \
#     --cs 8 \
#     --c_s 0.35 0.5 0.65 \
#     --log_name log_1030_sp_noise.txt \
#     --store_image;