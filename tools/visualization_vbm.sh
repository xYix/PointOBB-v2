python tools/visualize.py varmap \                                                                                                              
      configs/pointobbv2/train_cpm_vpd_point_dotav10.py \                                                                                         
      work_dirs/vpd_dotav10/epoch_2.pth \                                                                                                         
      --images P0000__1024__0___0.png \                                                                                                         
      --levels 0 1 \                                                                                                                              
      -o vis_output/varmap --device cuda:1 