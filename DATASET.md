# Dataset Preparation

## Setup Instructions

1. Download the **CIFAR-10 Python version** from the [official CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html).
2. Unzip the downloaded archive into the project directory.
3. Run the preparation script:

   ```bash
   python prepare_cifar10.py
   ```

   This generates the `cifar-10/` directory containing the training, validation, and test sets, organized into class-specific folders.

## Dataset Distribution

The dataset contains 60,000 images across 10 classes, split into 40,000 training, 10,000 validation, and 10,000 test images.

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Airplane | 4,000 | 1,000 | 1,000 | 6,000 |
| Automobile | 4,000 | 1,000 | 1,000 | 6,000 |
| Bird | 4,000 | 1,000 | 1,000 | 6,000 |
| Cat | 4,000 | 1,000 | 1,000 | 6,000 |
| Deer | 4,000 | 1,000 | 1,000 | 6,000 |
| Dog | 4,000 | 1,000 | 1,000 | 6,000 |
| Frog | 4,000 | 1,000 | 1,000 | 6,000 |
| Horse | 4,000 | 1,000 | 1,000 | 6,000 |
| Ship | 4,000 | 1,000 | 1,000 | 6,000 |
| Truck | 4,000 | 1,000 | 1,000 | 6,000 |
| **Total** | **40,000** | **10,000** | **10,000** | **60,000** |