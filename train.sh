nohup python3 -u train.py \
	--architecture shufflenet_v2_x1_0 \
	--num-epochs 90 \
	--num-workers 8 \
	--image-size 128 \
	--batch-size 896 \
	--optimizer Adam \
	--lr 1e-3 \
	--wd 4e-5 \
	--run-name "shufflenet-batch896-workers8" \
	--dataset-root /Vol1/dbstore/datasets/ImageNet \
> train-log.txt 2>&1
