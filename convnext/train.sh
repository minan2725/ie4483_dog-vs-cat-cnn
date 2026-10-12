python3 convnext/train_convnext_cifar10.py --train-per-class 2000 --lr 0.0001 --epochs 20 --seed 42
python3 convnext/train_convnext_cifar10.py --train-per-class 4000 --lr 0.0001 --epochs 20 --seed 1
python3 convnext/train_convnext_cifar10.py --train-per-class 4000 --lr 0.0001 --epochs 20 --seed 42
python3 convnext/train_convnext_cifar10.py --train-per-class 4000 --lr 0.0001 --epochs 20 --seed 42 --no-augment
python3 convnext/train_convnext_cifar10.py --train-per-class 4000 --lr 0.0001 --epochs 20 --seed 42 --freeze-backbone
python3 convnext/train_convnext_cifar10.py --train-per-class 4000 --lr 0.0001 --epochs 20 --seed 42 --scratch
python3 convnext/train_convnext_cifar10.py --train-per-class 4000 --lr 0.0003 --epochs 20 --seed 42
python3 convnext/train_convnext_cifar10.py --train-per-class 4000 --lr 0.00005 --epochs 20 --seed 42