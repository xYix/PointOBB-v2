CUDA_VISIBLE_DEVICES=1
# nohup
nohup bash tools/dist_train.sh configs/pointobbv2/train_cpm_vpd_point_dotav10.py \
    2 \
    --work-dir work_dirs/stride4 \
    &> work_dirs/stride4/nohup.log &