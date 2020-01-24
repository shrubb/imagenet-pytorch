python3 -um torch.distributed.launch --nproc_per_node=8 --master_port=5678 train.py --world_size=8 \
	--architecture shufflenet_v2_x1_0 \
	--num-epochs 90 \
	--num-workers 16 \
	--image-size 224 \
	--batch-size 512 \
	--optimizer Adam \
	--lr 1e-3 \
	--wd 4e-5 \
	--run-name "shufflenet-8gpu-batch4096-workers8" \
	--dataset-root /Vol1/dbstore/datasets/ImageNet \