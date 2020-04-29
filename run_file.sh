source ~/anaconda2/etc/profile.d/conda.sh
conda activate meitar

python -m torch.distributed.launch --nproc_per_node=1 main.py --dump_path ./exp/deepercluster/ --data_path /vildata/meitarr/yfcc100m-downloader_old/data_old/images/ --pretrained ./downloaded_models/deepercluster/ours.pth --size_dataset 2048 --workers 8 --sobel true --lr 0.1 --wd 0.00001 --nepochs 10 --batch_size 8 --reassignment 3 --dim_pca 256 --super_classes 1 --rotnet false --k 512 --warm_restart false --use_faiss true --niter 10 --dist-url file:///home/meitarr/file 

