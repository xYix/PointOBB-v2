python tools/visualize.py cpm \
      configs/pointobbv2/train_cpm_vpd_point_dotav10.py \
      work_dirs/cpm_dotav10/epoch_1_new.pth \
      --images P0001__1024__228___2472.png \
      --levels 0 1 \
      -o vis_output/cpm_fix --device cuda:1
      # checkpoints/train_cpm_epoch_6.pth \