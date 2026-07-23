# five_car_parts_semantic_segmentation
A project of semantic segmentation on five car parts

##
# Class-imbalance check



## Use TensorBoard to monitor Training process
1. Add tensorboard to your dependencies:

```bash
uv add tensorboard
```
2. Run training as normal — it now writes logs to runs/ (configurable via --log_dir):

```bash
uv run python train.py --images_dir data/train/images --masks_dir data/train/masks --pad_to_square
```
3. In a separate terminal, launch the dashboard:

```bash
uv run tensorboard --logdir results/training_outputs/training_runs
```

## Resume training
```bash
uv run python semantic_segmentation_five_car_parts/train.py --images_dir data/train/images --masks_dir data/train/masks     --resume_from results/training_outputs/training_run_initial/best_model.pth     --epochs 40 --pad_to_square 
```


## Inference
```bash
uv run python semantic_segmentation_five_car_parts/inference.py --input data/test/images --output data/test/masks --config_path outputs/config_2026-07-23_16-22-38.yaml
```

Check inference results
```bash
uv run python semantic_segmentation_five_car_parts/check_predictions.py --images_dir data/test/images/ --masks_dir data/test/masks/ --output_dir data/test/check_predictions
```


## Notes
To export `requirements.txt` through uv
```bash
uv export --format requirements-txt --no-hashes -o requirements.txt
```

document the nvidia-smi / driver-version caveat and the Blackwell/cu128 requirement explicitly in the README, a

used TensorBoard for local-only training monitoring, consistent with the dataset confidentiality requirement

### Verify PyTorch to GPU
```bash
uv run python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('compute capability:', torch.cuda.get_device_capability())  # expect (12, 0)
t = torch.tensor([2.0]).cuda()
print('real GPU compute test:', (t * t).item())  # this is the test that actually matters
"
```
If that last line runs without a `no kernel image error` and prints `4.0`, you're genuinely set up correctly — not just superficially.


Take the best model from results/training_outputs/training_run_2026-07-23_19-15-23/best_model.pth with "val_mIoU": 0.7298329935732258






### Implementation steps
1. check class balance with class_statistics_report, to get image numbers and pixel numbers of each class
1. crop training images and masks to remove background parts (keep some, but remove unnecessary large amounts) that are unnecessary for training
1. split train and validation sets (90/10) considering class distribution
    a. categorize images according to class, in each class take 90% for training and 10% for validation
1. image augmentation (albumentations)
1. start training using TensorBoard to monitor training process
1. use the best model weights with the highest val_mIoU to run inference
1. validate inference results by visually check the mask quality



## Lessons-learned
1. (door handles) Plausible, genuinely good explanation: Door handles likely have strong local visual contrast (different material/color against a painted door panel) and a fairly consistent shape/location — easy for a CNN to localize precisely once detected. Meanwhile, Front Door/Rear Door/Fender boundaries are often subtle body-panel creases against each other, not against a clean background — genuinely harder to delineate with pixel-precision even though they're "big, easy" classes on paper. Your Dice+CE weighting strategy may have simply worked exactly as designed here.

2. (go on training until platenau is reached) Epoch 59 of a 60-epoch budget — this means early stopping (patience=12) never triggered, so val mIoU was still improving (or at least hadn't plateaued for 12 straight epochs) right up to your cap. That's actually good news framed one way ("didn't need early stopping, model kept legitimately improving") but also means you may be leaving performance on the table — if you'd had --epochs 100, it's plausible mIoU keeps climbing further. Worth checking the TensorBoard IoU/mean_val curve's trend in the last 10-15 epochs: if it's still trending up (not flattened), that's your strongest lever for a meaningful improvement with minimal extra effort — just extend the epoch budget and rerun.

3. (less data points from front fender with the weakest per class IoU)

6. consider tunning hyper-parameter

4. consider class distribution in train/val splitting

5. consider ResNet50 with more dimensions

