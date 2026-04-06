python tools/visualize.py cpm \
      configs/pointobbv2/train_cpm_vpd_point_dotav10.py \
      checkpoints/train_cpm_epoch_6.pth \
      --images P0001__1024__228___2472.png \
      --levels 0 1 \
      -o vis_output/cpm --device cuda:1