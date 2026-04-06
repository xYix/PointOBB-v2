CUDA_VISIBLE_DEVICES=0,1
# nohup
nohup bash tools/dist_train.sh configs/pointobbv2/train_cpm_vpd_point_dotav10.py 2 &> work_dirs/vpd_dotav10/nohup.log &