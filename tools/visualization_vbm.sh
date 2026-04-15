python tools/visualize.py varmap \
      configs/pointobbv2/train_cpm_vpd_point_dotav10.py \
      work_dirs/train_cpm_vpd_point_dotav10/epoch_1.pth \
      --images P0001__1024__228___2472.png \
      --levels 0 1 \
      -o vis_output/varmap --device cuda:1