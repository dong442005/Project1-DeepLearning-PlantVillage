# Plant Disease Classification Using Deep Learning

## Overview

A university Deep Learning project for classifying PlantVillage leaf images into **38 disease and healthy-leaf categories**. The project compares two CNNs trained from scratch with two ImageNet-pretrained models:

- **Simple CNN:** a LeNet-inspired baseline with three convolutional stages.
- **Complex CNN:** four convolutional blocks with batch normalization, max pooling, and dropout.
- **ResNet50:** transfer learning with a residual backbone.
- **MobileNetV2:** transfer learning with a lightweight backbone.

## Dataset

Images are downloaded from the [PlantVillage Kaggle dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset). The [original dataset repository](https://github.com/spMohanty/PlantVillage-Dataset) provides further context.

- **Split:** approximately 70% training, 15% validation, and 15% test, applied within each class with `random_state=42`.
- **Input size:** 224 × 224 RGB images.
- **Preprocessing:** Z-score normalization for the custom CNNs; backbone-specific `preprocess_input` for ResNet50 and MobileNetV2.
- **Augmentation:** rotation, shifts, zoom, and horizontal flips during training; the transfer learning models also use shear.

The dataset is excluded from Git. Preserve the prepared partition for consistent checkpoint comparisons: source filenames are not sorted before splitting, so partitions may differ between machines.

## Project Structure

```text
Project1-DeepLearning-PlantVillage/
├── src/
│   ├── data_prep.py           # Download and split the dataset
│   ├── simple_cnn.py          # Train and evaluate Simple CNN
│   ├── complex_cnn.py         # Train Complex CNN
│   ├── resnet_model.py        # Train and evaluate ResNet50
│   ├── transfer_learning.py   # Train and evaluate MobileNetV2
│   ├── predict.py             # Evaluate saved models or predict one image
│   └── model_loader.py        # Saved-model compatibility handling
├── notebooks/                # Dataset exploration and model experiments
├── outputs/                  # Per-model figures and comparison reports
├── data/                     # Local train/, val/, and test/ directories
├── models/                   # Local trained checkpoints
├── requirements.txt
└── README.md
```

## Installation & Usage

Use **Python 3.11** and run commands from the repository root. CPU execution is supported; GPU acceleration is optional.

**Clone and install — macOS / Linux:**

```bash
git clone https://github.com/dong442005/Project1-DeepLearning-PlantVillage.git
cd Project1-DeepLearning-PlantVillage
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.11 -m venv .venv311` and activate with `.\.venv311\Scripts\Activate.ps1` in PowerShell, then run the same dependency installation command. Training dependencies pin TensorFlow and Keras to **2.15.0**.

**Prepare the dataset:**

```bash
python src/data_prep.py
```

This creates `data/train/`, `data/val/`, and `data/test/`, with one directory per category. **Rerunning the script deletes the existing `data/` directory.** It expects `plantvillage dataset/color` within the download; check the downloaded layout if that path is absent.

**Train each model:**

```bash
python src/simple_cnn.py
python src/complex_cnn.py
python src/resnet_model.py
python src/transfer_learning.py
```

All four use batch size 32. Simple CNN defaults to 20 epochs; Complex CNN allows up to 30. Both use early stopping on validation loss with patience 5. ResNet50 and MobileNetV2 each use 10 frozen-backbone epochs followed by 10 fine-tuning epochs.

Simple CNN also supports `--epochs 30` and `--smoke-test`; the other scripts do not implement these options. Complex CNN saves `models/complex_cnn_best.keras` and stops if that file already exists. The other scripts save their `.h5` models in the repository root. Figures and reports are stored under the corresponding model directory in `outputs/`; Complex CNN test evaluation is available in `notebooks/complex_cnn.ipynb`.

**Evaluate saved checkpoints without retraining:**

The local checkpoints include Keras 3 files. Create a separate inference environment:

```bash
python3.11 -m venv .venv-keras3
source .venv-keras3/bin/activate
python -m pip install tensorflow==2.20.0 keras==3.13.2 numpy==1.26.4 scikit-learn matplotlib pillow
python src/predict.py
```

On Windows, create this environment with `py -3.11` and activate `.\.venv-keras3\Scripts\Activate.ps1`. Before prediction, place `simple_cnn_model.h5`, `complex_cnn_best.keras`, `resnet50_finetuned_model.h5`, and `mobilenetv2_finetuned_model.h5` in `models/`. Checkpoints are excluded from Git; a fresh clone requires training them or obtaining the files separately.

Keep `IMAGE_PATH = None` and `TEST_STEPS = None` in `src/predict.py` for the full-test comparison. Set `IMAGE_PATH` to an existing image path for a single-image demonstration. Results are written to `outputs/comparison/`; CSV files remain local because they are ignored.

## Results

The comparison below uses **8,162 test images**, with metrics verified against the [checkpoint evaluation reports](outputs/comparison/).

| Model                                                    | Test Accuracy | Macro F1-score |
| -------------------------------------------------------- | ------------: | -------------: |
| [Simple CNN](outputs/comparison/simple_cnn_report.txt)   |        95.71% |         0.9491 |
| [Complex CNN](outputs/comparison/complex_cnn_report.txt) |        95.72% |         0.9403 |
| [ResNet50](outputs/comparison/resnet50_report.txt)       |        99.04% |         0.9891 |
| [MobileNetV2](outputs/comparison/mobilenetv2_report.txt) |        96.17% |         0.9575 |

ResNet50 has the highest accuracy and macro F1 in this comparison. The custom CNNs have similar accuracy, while Simple CNN has higher macro F1. These results describe the evaluated checkpoints and do not establish performance on field photographs.

Earlier training reports and figures may represent different runs. In particular, the older Simple CNN report records 95.03% accuracy; the table uses the common checkpoint evaluation above.

## Team

| Member      | Contribution                  |
| ----------- | ----------------------------- |
| Phương Đông | Data preparation and ResNet50 |
| Vân Thư     | Simple CNN                    |
| Thu Trang   | Complex CNN                   |
| Việt Hằng   | MobileNetV2                   |
