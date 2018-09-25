nohup python3 -u train.py \
	--arch mobilenetv2 \
	--epochs 48 \
	--workers 16 \
	--batch-size 1536 \
	--optimizer Adam \
	--lr 1e-3 \
	--wd 4e-5 \
	--run-name "superconvergence-mobilenetv2-wd4e-5" \
	--resume checkpoint.pth.tar \
	~/Datasets/ImageNet \
> train-log.txt 2>&1

