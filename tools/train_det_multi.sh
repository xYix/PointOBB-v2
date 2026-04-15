CUDA_VISIBLE_DEVICES=0,1
PORT=29801
nohup bash tools/dist_train.sh configs/pointobbv2/redet_dotav10.py 2 \
    --work-dir work_dirs/det_vpd_sigma \
    &> work_dirs/det_vpd_sigma/nohup.log &
