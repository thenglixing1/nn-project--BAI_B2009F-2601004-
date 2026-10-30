# Malaysian Dish Classification with a Custom CNN

**Course:** BAI13143 Neural Network Basics (Individual Project)
**Student:** [Your Name] ([Student ID])

A convolutional neural network, designed from scratch in PyTorch, that recognises five Malaysian dishes from photos: **laksa, mixed rice, nasi lemak, roti canai and satay**. Training and testing each run in their own Docker container, so the results can be reproduced on a different machine.

## Dataset

- **Malaysia Food 11** by Kar Keng Chan, Kaggle: https://www.kaggle.com/datasets/karkengchan/malaysia-food-11
- Licence: "Database: Open Database, Contents: © Original Authors". Used for academic coursework only.
- 5 of the 11 classes are used. After removing 1 corrupt image and 19 duplicate images, 4,980 images remain, split 70/15/15 (3,485 train / 745 validation / 750 test).
- **The images are not in this repository.** `models/splits.csv` lists exactly which images belong to which split. Images that are not listed are ignored.

### Getting the data
1. Download the zip from the Kaggle page above and unzip it.
2. Copy only these five folders into `data/` (keep the folder names):

```
data/
├── laksa/
├── mixed_rice/
├── nasi_lemak/
├── roti_canai/
└── satay/
```

## Project structure

```
├── data/                    dataset (mounted into containers, not copied into images)
├── models/                  saved models, config.json, splits.csv (shared via volume)
│   └── final/               final model used by the test container
├── figures/                 plots, experiment_log.csv, cleaning_log.csv
├── src/
│   ├── explore_data.py      data exploration and figures
│   ├── clean_and_split.py   removes corrupt/duplicate images, writes models/splits.csv
│   ├── dataset.py           loading and preprocessing (shared by train/test/predict)
│   ├── model.py             the CNN (FoodNet) and parameter table
│   ├── train.py             training, curves, experiment log
│   ├── test.py              evaluation: accuracy, precision, recall, F1, confusion matrix
│   └── predict.py           predict new photos
├── Dockerfile.train
├── Dockerfile.test
├── requirements-train.txt
└── requirements-test.txt
```

## Requirements

- Docker Desktop (CPU only, no GPU needed)
- Run the commands below from the project root. They use `$(pwd)`, which works on macOS, Linux and Git Bash. In PowerShell use `${PWD}` instead.

## 1. Train in a container

```bash
docker build -f Dockerfile.train -t nn-train .
docker run --rm \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/figures":/app/figures \
  nn-train
```

The default run trains 50 epochs with data augmentation and saves the model, `config.json` (image size, normalisation values, class names), the training history and the loss/accuracy curves to `models/docker_final/` and `figures/`. This takes about an hour on a CPU. Other settings can be passed after the image name, for example `nn-train --name my_run --epochs 10`.

## 2. Test in a separate container

The test image contains only what is needed for inference. It loads `models/final/` and evaluates it on the 750 test images.

```bash
docker build -f Dockerfile.test -t nn-test .
docker run --rm \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/models":/app/models \
  nn-test
```

It prints accuracy, per-class precision, recall and F1, the confusion matrix and a simple baseline (always predicting the most common class). It also saves `test_metrics.json`, `test_confusion_matrix.csv` and `test_wrong_predictions.csv` into `models/final/`. **The test set is meant to be used once, for the final model.**

## 3. Predict a new photo

Put a photo in a folder called `samples/`, then:

```bash
docker run --rm \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/samples":/app/samples \
  nn-test src/predict.py --model-dir models/final samples/my_photo.jpg
```

It prints the predicted dish and the top-3 probabilities.

## Reproducibility

- Library versions are pinned in `requirements-train.txt` and `requirements-test.txt`.
- The random seed is fixed (42) and everything runs on the CPU. Two runs with the same settings on the same machine gave identical numbers.
- The train/validation/test split is saved in `models/splits.csv`, so both containers use the same images.
- Small numerical differences can appear between different CPUs or operating systems.

## Experiments

All runs use the same seed and change one thing relative to the baseline. Validation results (best epoch by validation loss):

| Run | What changed | Best epoch | Val loss | Val accuracy |
|---|---|---|---|---|
| baseline | – | 10 | 0.747 | 72.8% |
| exp1_augment | data augmentation | 25 | 0.643 | 75.6% |
| exp2_lr | learning rate 1e-3 to 3e-4 | 10 | 0.822 | 69.7% |
| exp3_aug_long | augmentation, 50 epochs | 50 | 0.546 | 81.7% |

Full log: `figures/experiment_log.csv`. Curves: `figures/*_curves.png`.

## Final result (test set)

[Fill in from `models/final/test_metrics.json` after the final test run: accuracy, macro F1, comparison with the target and with the majority-class baseline.]

## Running without Docker (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
python src/train.py --name baseline --note "Baseline"
python src/train.py --name exp1_augment --augment --note "Add data augmentation"
python src/model.py        # prints the layer table and parameter counts
```

Tested with Python 3.14, PyTorch 2.14.1 and torchvision 0.29.1.

## Generative AI declaration

See the "Reflection" section of the report.