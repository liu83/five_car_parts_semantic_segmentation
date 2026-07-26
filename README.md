# five_car_parts_semantic_segmentation
A project of semantic segmentation on five car parts

## Installation

### Using `uv`
It is recommanded to use `uv` to install the project dependencies. Make sure to install `uv` beforehand.

#### Install `uv`
One-liner (official installer):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then either restart your shell or reload your PATH:
```bash
source $HOME/.local/bin/env
```
Verify by:
```bash
uv --version
```
#### Install dependencies

Given existing `pyproject.toml` and `uv.lock`, run:
```bash
uv sync
```
The dependencies will be installed automatically.

#### Verify PyTorch to GPU
After installing dependencies, make sure that GPU is usable with installed PyTorch and CUDA version. Run:
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
If the last line runs without a `no kernel image error` and prints `4.0`, you are set up correctly.



### Using `pip`
Dependencies can also be installed using `pip` based on the existing `requirements.txt`. After creating a virtual environment, activate that `.venv` and run:
```bash
pip install -r requirements.txt
```


## Data Pre-processing

### Compute Class Statistics
The script `class_statistics.py` loads the image masks from train dataset to calculate class distribution/statistics of the train dataset for a better overview of the train dataset/class imbalance.

Command to run the script:
```bash
uv run python semantic_segmentation_five_car_parts/class_statistics.py
```

### Crop Images
Script `bbox_utils.py` includes functions to compute bounding box that covers only car parts and background nearby, and excludes other surrounding background. This functionality will be used in creating PyTorch Dataset for training.

Script `check_crops.py` visualizes this cropping functionality on several sample images. Example command:
```bash
uv run python semantic_segmentation_five_car_parts/check_crops.py \
    --images_dir data/train/images/ \
    --output_dir data/cropped_samples/ \
    --num_samples 6
```

Script `crop_dataset.py` computes all cropped images and masks in the training dataset. It saves them as well for further checks. Example command:
```bash
uv run python semantic_segmentation_five_car_parts/crop_dataset.py \
    --images_dir data/train/images/ \
    --masks_dir data/train/masks/ \
    --output_dir data/cropped
```

### Image Augmentation
For a robust training result and to enrich training dataset with limited amount, image augmentation is applied within `train.py`. Script `check_augmentations.py` visualizes augmentation results by running:
```bash
uv run python semantic_segmentation_five_car_parts/check_augmentations.py \
    --images_dir data/train/images \
    --masks_dir data/train/masks \
    --output_dir data/aug_checks/
```
Augmentated images will be saved in the `output_dir`.

## Training
Run following command to start training the model:
```bash
uv run python semantic_segmentation_five_car_parts/train.py \
    --images_dir data/train/images \
    --masks_dir data/train/masks \
    --pad_to_square
```
See script `train.py` for more possible parameter configuration.

### Monitor Training Process using TensorBoard
During training, launch the dashboard in a separate terminal:
```bash
uv run tensorboard --logdir results/training_outputs/training_runs
```

### Resume Training
To resume a Training using an existing weights, run command:
```bash
uv run python semantic_segmentation_five_car_parts/train.py \
    --images_dir data/train/images \
    --masks_dir data/train/masks \
    --resume_from results/training_outputs/training_run_initial/best_model.pth \
    --epochs 40 \
    --pad_to_square 
```

## Inference
After training is finished, script `inference.py` can be called to run inference on test dataset using trained model:
```bash
uv run python semantic_segmentation_five_car_parts/inference.py \
    --input data/test/images \
    --output_dir results_submission/predictions/masks/ \
    --config_path results_submission/training_run_02_resumed/train_config_2026-07-23_19-15-23.yaml \
    --model_path results_submission/training_run_02_resumed/best_model.pth \
    --report_path results_submission/training_run_02_resumed/inference_report.yaml
```
Inference results (image masks) are saved in the output directory `results_submission/predictions`.

### Visualize Inference Results
Script `check_predictions.py` visualizes the inference results by overlaying the predicted masks on the images. Run:
```bash
uv run python semantic_segmentation_five_car_parts/check_predictions.py \
    --images_dir data/test/images/ \
    --masks_dir results_submission/predictions/masks/ \
    --output_dir results_submission/predictions/masked_images
```
Images with predicted masks can be found in `results_submission/predictions/masked_images`.



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



