# Read labels of CIFAR10 test set
import os
import pandas as pd

test_dir = "cifar-10/test"
class_names = sorted(os.listdir(test_dir))
class_names = [name for name in class_names if os.path.isdir(os.path.join(test_dir, name))]

data = []
for label, class_name in enumerate(class_names):
    class_dir = os.path.join(test_dir, class_name)
    for filename in os.listdir(class_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            data.append({"filename": os.path.basename(filename), "label": label})

pd.DataFrame(data).sort_values("filename").to_csv("cifar_ground_truth.csv", index=False)