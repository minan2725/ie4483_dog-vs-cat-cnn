# Dogs vs. Cats training protocol

This defines a comparable reference run for each architecture while leaving model-specific experiments in each model folder. The ConvNeXt, ResNet50, and VGG16 entry points use `dogcat_training.py` for the common CLI, training loop, checkpoint selection, run history, and optional test prediction. Everyone trains on their own machine; the ResNet50 plan in section 2 is a suggestion so that the three models can be compared, not a limit on what anyone else may run.

## 1. Common reference run

| Item               | Convention                                                                                                                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data               | Use`preprocessing.get_loaders()` and all 19,981 clean training images (19 duplicates of val/test images removed); keep all 5,000 validation images out of training. No random subsets in reported runs. |
| Labels             | `cat=0`, `dog=1`.                                                                                                                                                                                     |
| Input              | RGB, 224 x 224 (fixed: the size the ImageNet weights were trained at); use the shared training and evaluation transforms.                                                                                 |
| Initialization     | Start from ImageNet-pretrained weights and fine-tune all layers; record exceptions.                                                                                                                       |
| Loss and optimizer | Cross-entropy and AdamW, weight decay`1e-4`, reference learning rate `1e-4`.                                                                                                                          |
| Batch size         | 32. Any other value is passed with`--batch-size` and must be mentioned in the report.                                                                                                                   |
| Epochs             | 10                                                                                                                                                                                                        |
| Checkpoint         | Highest validation accuracy; ties are broken by the lower validation loss.                                                                                                                                |
| Randomness         | Seed 42 for all experiments; the best configuration is repeated with seeds 1 and 2. Report mean and sample standard deviation over the 3 seeds.                                                           |
| Test               | Generate all 500`id,label` rows after selection (`--predict-test`). Consult `test_answer.csv` only for final evaluation, never for tuning.                                                          |

## 2. Suggested experiments (what ResNet50 runs; match it so the models are comparable)

| # | Runs                                                                                     | What it lets us conclude                                                                                                               |
| - | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Fine-tuning at learning rate 1e-5, 1e-4, 1e-3 (3 runs)                                   | Which learning rate to use and why: too small converges slowly, too large damages the pretrained features. The winner is the baseline. |
| 2 | Baseline without augmentation (1 run)                                                    | Whether augmentation is kept, read from validation accuracy and the train-validation gap.                                              |
| 3 | Best configuration of 1 and 2 with seeds 1 and 2 (2 runs)                                | Mean and std of the final model, and the noise level: a difference between runs smaller than about 2 std is not evidence.              |
| 4 | Training from scratch at lr 1e-3, same batch size and epoch limit (1 run, ResNet50 only) | How much the pretrained weights contribute. If training is still improving at the epoch limit, the gap is a lower bound.               |

Not run, on purpose: input size (fixed at 224), frozen backbone (it only tells us that updating the backbone helps), and a reduced-data run (a difference of a few images in 5,000 cannot be separated from seed noise).
