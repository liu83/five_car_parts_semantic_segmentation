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
uv run python3 train.py --images_dir data/images --masks_dir data/masks
```
3. In a separate terminal, launch the dashboard:

```bash
uv run tensorboard --logdir runs
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




### Implementation steps
1. check class balance with class_statistics_report, to get image numbers and pixel numbers of each class
1. crop training images and masks to remove background parts (keep some, but remove unnecessary large amounts) that are unnecessary for training
1. split train and validation sets (90/10) considering class distribution
    a. categorize images according to class, in each class take 90% for training and 10% for validation
1. image augmentation (albumentations)



