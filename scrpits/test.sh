python prediction.py \
--ckpt_path ./checkpoints/40epoch \
--results_path ./results/40epoch \
--network Unet \
--epochs 3 \
--batch_size 8 \
--opt 1 \
--split val \
--weights weights.36-0.06-0.76.hdf5
