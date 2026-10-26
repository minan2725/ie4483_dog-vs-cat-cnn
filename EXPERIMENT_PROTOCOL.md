# Proposed Dogs vs. Cats training protocol

This defines a comparable reference run for each architecture while leaving model-specific experiments in each model folder. The ConvNeXt, ResNet50, and VGG16 entry points now use `dogcat_training.py` for the common CLI, training loop, checkpoint selection, run history, and optional test prediction. Older run records remain unchanged; the full-data ConvNeXt result predates this refactor but used the same reference settings.

## Common reference run

| Item | Convention |
| --- | --- |
| Data | Use `preprocessing.get_loaders()` and all 19,981 clean training images; keep all 5,000 validation images out of training. |
| Labels | `cat=0`, `dog=1`. |
| Input | RGB, 224 x 224; use the shared training and evaluation transforms. |
| Initialization | Start from ImageNet-pretrained weights and fine-tune all layers; record exceptions. |
| Loss and optimizer | Cross-entropy and AdamW with learning rate `1e-4`, weight decay `1e-4`. |
| Schedule | Cosine decay over five epochs, batch size 32. |
| Randomness | Seed 42 for the reference run. Add seed 1 when comparing architectures; report mean and sample standard deviation only when both runs exist. |
| Checkpoint | Select by highest validation accuracy; break ties using lower validation loss. |
| Test | Generate all 500 `id,label` rows after selection. Consult `test_answer.csv` only for final evaluation, never for tuning. |

If a model cannot use a common setting (for example, a smaller batch size for memory), record the deviation. A tuned run can use different hyperparameters, but present it separately from the common reference run so readers can tell what was held fixed.

## Run record and report

Keep each run in its model's `runs/<model>_lr<rate>_n19981_e5_s<seed>/` folder. The shared pipeline saves `history.json` with the arguments, device, PyTorch version, actual train/validation counts, best epoch, best validation accuracy, and per-epoch train/validation loss, accuracy, and time. Keep `best.pt` local because weights are ignored by Git. Optional test prediction writes `submission.csv` and `test_probs.csv`; save final test metrics separately after choosing the run. Recording a Git commit and dataset fingerprint remains a future improvement.

From the repository root, run `.venv/bin/python <model>/train_<model>.py` for the common full-data reference settings. Use `--train-per-class` and `--val-per-class` only for explicitly labelled subset experiments, `--out-dir` for a separate run folder, and `--predict-test` only after deciding to generate predictions from the validation-selected checkpoint. The training command writes predictions without reading test labels; final test metrics are a separate evaluation step.

For the group report, use one compact table with model, train/validation counts, pretrained status, input size, epochs, batch size, learning rate, seed, best validation accuracy, and test accuracy. State how the checkpoint was chosen and describe one or two prediction examples. Compare validation results only when splits and evaluation transforms match. A single seed is a result, not an estimate of run-to-run variation.

This protocol covers Dogs vs. Cats only. CIFAR-10 has different classes, splits, and scripts; report it separately.
