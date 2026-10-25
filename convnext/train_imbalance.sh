python3 convnext/train_convnext_imbalance.py --imbalance 0.05 1 1 0.1 1 1 0.05 1 1 0.8 --lr 0.0001 --epochs 20
python3 convnext/train_convnext_imbalance.py --imbalance 0.05 1 1 0.1 1 1 0.05 1 1 0.8 --lr 0.0001 --epochs 30 --imbalance-train weighted_loss
python3 convnext/train_convnext_imbalance.py --imbalance 0.05 1 1 0.1 1 1 0.05 1 1 0.8 --lr 0.0001 --epochs 30 --imbalance-train focal_loss
python3 convnext/train_convnext_imbalance.py --imbalance 0.05 1 1 0.1 1 1 0.05 1 1 0.8 --imbalance-train resampling --lr 0.0001 --epochs 20
python3 convnext/train_convnext_imbalance.py --imbalance 0.05 1 1 0.1 1 1 0.05 1 1 0.8 --imbalance-train synthetic --lr 0.0003 --epochs 30
python3 convnext/train_convnext_imbalance.py --imbalance 0.05 1 1 0.1 1 1 0.05 1 1 0.8 --imbalance-train synthetic --lr 0.00005 --epochs 30
python3 convnext/train_convnext_imbalance.py --imbalance 0.05 1 1 0.1 1 1 0.05 1 1 0.8 --imbalance-train synthetic --lr 0.0001 --epochs 30