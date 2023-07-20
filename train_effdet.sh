python train_effdet.py data/Supermarket-Empty-Shelf-Detector--3coco --model \
efficientdet_d0 -b 16 --amp --lr .008 --sync-bn --opt fusedmomentum \
--warmup-epochs 1 --model-ema --model-ema-decay 0.9966 --epochs 1 --num-classes 1 --pretrained