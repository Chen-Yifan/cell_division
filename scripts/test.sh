python prediction.py \
--ckpt_path ./checkpoints/480_40cm_noresize_try \
--results_path ./results/480_40cm_noresize_try \
--network Unet \
--epochs 26 \
--batch_size 16 \
--opt 1 \
--split test
