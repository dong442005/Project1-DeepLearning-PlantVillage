# PlantVillage Leaf Disease Classification

This repository contains the implementation of a Deep Learning course project on multiclass plant leaf disease classification. The study compares two custom convolutional neural networks (CNNs) with two transfer learning models using a common PlantVillage data partition. The objective is to examine their predictive performance under the same evaluation conditions.

## Models and contributions

| Contributor | Responsibility | Model |
| --- | --- | --- |
| Phương Đông | Data preparation and transfer learning | ResNet50 |
| Vân Thư | Baseline architecture and evaluation | Simple CNN |
| Thu Trang | Deep custom architecture and evaluation | Complex CNN |
| Việt Hằng | Transfer learning and fine-tuning | MobileNetV2 |

The Simple CNN is a LeNet-5-inspired baseline with three convolutional stages followed by fully connected layers. The Complex CNN uses four convolutional blocks with batch normalization and dropout. ResNet50 and MobileNetV2 use ImageNet initialization, followed by training of the classification head and fine-tuning of selected backbone layers.

## Dataset and experimental protocol

The project uses the color images from the [PlantVillage dataset distributed through Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset). The classification task comprises 38 disease and healthy-leaf categories.

The preparation script partitions each category into approximately 70% training, 15% validation, and 15% test images using `random_state=42`. All models must use the same prepared directories. The script does not explicitly sort the source filenames before splitting; retaining the prepared partition is therefore necessary for comparisons across runs or machines.

Images are resized to 224 × 224 pixels. Training uses image augmentation; validation and test images are processed without augmentation. The custom CNNs use Z-score normalization with training-set channel statistics. Each transfer learning model uses the preprocessing function associated with its backbone. Class indices follow the alphabetical order of category directory names.

The evaluation reports accuracy and macro-averaged precision, recall, and F1-score. Macro averaging assigns equal weight to each category and complements accuracy when category sizes differ. Test data should be reserved for final evaluation rather than model selection.

## Repository structure

```text
Project1-DeepLearning-PlantVillage/
├── data/                       # Local dataset: train/, val/, and test/
├── models/                     # Local saved model files
├── notebooks/                  # Exploratory analysis and model experiments
├── outputs/
│   ├── simple_cnn/             # Training figures and baseline results
│   ├── complex_cnn/            # Training figures and prediction analysis
│   ├── resnet50/               # Training figures and confusion matrix
│   ├── mobilenetv2/            # Training figures and confusion matrix
│   └── comparison/             # Evaluation reports and prediction CSV files
├── src/
│   ├── data_prep.py            # Download and partition the dataset
│   ├── simple_cnn.py           # Train and evaluate the baseline CNN
│   ├── complex_cnn.py          # Train the deep custom CNN
│   ├── resnet_model.py         # Train and evaluate ResNet50
│   ├── transfer_learning.py    # Train and evaluate MobileNetV2
│   ├── predict.py              # Load saved models and compare predictions
│   └── model_loader.py         # Handle saved-model compatibility
├── requirements.txt
├── .gitignore
└── README.md
```

Datasets, model files, CSV files, local documentation, and experimental weighted Complex CNN files are excluded from version control. The reported comparison includes only Simple CNN, Complex CNN, ResNet50, and MobileNetV2.

## Installation and data preparation

Run all commands from the repository root. For the TensorFlow 2.15 training configuration, create a separate Python 3.10 or 3.11 environment:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

To download and prepare the dataset:

```bash
python src/data_prep.py
```

The preparation script deletes an existing `data/` directory before rebuilding the partition. Run it only when a new partition is required; preserve the existing partition when evaluating saved models.

## Training

Run the training script for the required architecture:

```bash
python src/simple_cnn.py
python src/complex_cnn.py
python src/resnet_model.py
python src/transfer_learning.py
```

Each command starts a separate training run. Simple CNN saves its best validation-loss checkpoint as `simple_cnn_model.h5`. Complex CNN saves `models/complex_cnn_best.keras` and refuses to overwrite an existing checkpoint. The transfer learning scripts save their final fine-tuned models as `resnet50_finetuned_model.h5` and `mobilenetv2_finetuned_model.h5` in the repository root.

Move the three `.h5` files into `models/` before running the comparison script. Saved weights are distributed separately from the source code. The existing [ResNet50 model download](https://drive.google.com/file/d/1DooI4k3YiHRk8JXDBMMcKT2jk37YSW_T/view?usp=drive_link) is provided by the project team.

## Inference and model comparison

Inference reloads saved models and does not retrain them. The current saved models include Keras 3 files; use a separate environment for prediction:

```bash
python3.11 -m venv .venv-keras3
source .venv-keras3/bin/activate
pip install tensorflow==2.20.0 keras==3.13.2 numpy==1.26.4 scikit-learn matplotlib pillow
```

Prepare the following files:

```text
models/
├── simple_cnn_model.h5
├── complex_cnn_best.keras
├── resnet50_finetuned_model.h5
└── mobilenetv2_finetuned_model.h5
```

Retain the original category directory names in `data/train/` and `data/test/`. The inference script obtains the class order from the training directories and checks that the test category names match.

Run the comparison:

```bash
python src/predict.py
```

With `IMAGE_PATH = None` and `TEST_STEPS = None`, the script evaluates all four saved models on the complete test partition, prints a comparison table, and writes the following files to `outputs/comparison/`:

- `model_comparison.csv`: sample count, accuracy, macro precision, macro recall, and macro F1 for each model.
- `<model>_report.txt`: the classification report for each category.
- `<model>_predictions.csv`: image paths, true labels, and predicted labels.

To verify the workflow quickly, set `TEST_STEPS = 2` near the beginning of `src/predict.py`. This evaluates only the first two batches in directory order and does not provide a representative estimate of full-test performance. Restore `TEST_STEPS = None` for the final comparison.

For a single-image demonstration, set `IMAGE_PATH` to the path of an existing image. The script prints the predicted category and maximum softmax score for each model. This score is a confidence output, rather than a measure of prediction accuracy.

The MobileNetV2 file currently available to the team was saved with a newer Keras development version. `model_loader.py` handles specific inactive configuration fields on a temporary copy when required. The original model file and its stored weights are preserved.

For the oral examination, complete the full-test run in advance and retain the terminal containing the final comparison table and completion message.

## Interpretation and limitations

Metrics correspond to the particular saved models and local test partition used in a run. Training figures and earlier reports may describe different checkpoints; they should not be combined with a new comparison without verifying their provenance. High performance on PlantVillage images does not establish performance on field photographs with different backgrounds, lighting, or acquisition conditions.
