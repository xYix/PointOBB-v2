CUDA_VISIBLE_DEVICES=0,1
PORT=29816
# tools/dist_test1.sh work_dirs/redet_dotav10/redet_dotav10.py work_dirs/det_obbv2/epoch_1.pth 2
tools/dist_test1.sh configs/pointobbv2/redet_dotav10.py work_dirs/det_obbv2/epoch_1.pth 2