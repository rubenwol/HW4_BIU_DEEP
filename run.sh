CUDA_VISIBLE_DEVICES=0 python train.py glove/glove.6B.300d.txt 0.0004 300 0.1
CUDA_VISIBLE_DEVICES=1 python train.py glove/glove.6B.300d.txt 0.0004 300 0.1
CUDA_VISIBLE_DEVICES=2 python train.py glove/glove.6B.300d.txt 0.0004 300 0.1
CUDA_VISIBLE_DEVICES=3 python train.py glove/glove.840B.300d.txt 0.0004 300 0.1


