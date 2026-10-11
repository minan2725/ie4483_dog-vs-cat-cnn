python vgg16/train_vgg16.py --train-per-class 5000 --lr 0.001 --epochs 12 --seed 42
python vgg16/train_vgg16.py --train-per-class 5000 --lr 0.0001 --epochs 12 --seed 42 --img-size 128
python vgg16/train_vgg16.py --train-per-class 5000 --lr 0.0001 --epochs 12 --seed 42 --frozen
python vgg16/train_vgg16.py --train-per-class 5000 --lr 0.0001 --epochs 20 --seed 42 --scratch
python vgg16/train_vgg16.py --train-per-class 5000 --lr 0.0001 --epochs 12 --seed 42 --no-aug
python vgg16/train_vgg16.py --train-per-class 5000 --lr 1e-05 --epochs 12 --seed 42