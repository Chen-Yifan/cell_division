#!/bin/bash
python cv_main.py \
--date 9.24nonorm \
--name 128over5m6b_unet_Adadelta_der1_200w_30e \
--model unet \
--loss_function single \
--epoch 30 \
--augmentation 0 \
--mask_name all_masks_5m6b \
--weight 200 \
--frame_name all_frames_5m6b \
--optimizer 2 \
--k 2 \
--derivative 1
