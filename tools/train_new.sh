nohup python tools/train.py configs/pointobbv2/train_cpm_vpd_point_dotav10.py \
     --work-dir work_dirs/sigmafix \
     --gpu-ids 1 \
     > work_dirs/sigmafix/train.log 2>&1