# Proposed Dogs vs. Cats training protocol

This is a shared reporting convention, not a new training framework. It defines a comparable reference run for each architecture while leaving model-specific experiments in each model folder. The ConvNeXt full-data run follows this convention; the older ResNet50 and VGG16 scripts still need to be updated for the current `preprocessing.get_loaders()` interface before they can run it.

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

Keep each run in its model's `runs/<model>_lr<rate>_n19981_e5_s<seed>/` folder. Save `history.json` with the arguments, device, package versions, actual train/validation counts, best epoch, best validation accuracy, and per-epoch train/validation loss, accuracy, and time. Keep `best.pt` local because weights are ignored by Git. Save the final `submission.csv`, probability file, and final metrics next to the run record. Where practical, record a Git commit and dataset fingerprint too; those fields are proposed, not present in every current script.

For the group report, use one compact table with model, train/validation counts, pretrained status, input size, epochs, batch size, learning rate, seed, best validation accuracy, and test accuracy. State how the checkpoint was chosen and describe one or two prediction examples. Compare validation results only when splits and evaluation transforms match. A single seed is a result, not an estimate of run-to-run variation.

This protocol covers Dogs vs. Cats only. CIFAR-10 has different classes, splits, and scripts; report it separately.
