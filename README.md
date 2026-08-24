# Bone X-Ray Abnormality Detection

A deep learning-based project for detecting abnormalities in bone X-ray images.

## Overview

This project focuses on binary classification of bone X-ray images using deep learning. Several training strategies were developed and evaluated to improve the detection of abnormal X-ray images.

The project also includes Grad-CAM based visual analysis to provide an interpretable view of the regions that influence model predictions.

## Project Structure

```text
bone-xray-abnormality-detection/
│
├── src/
│   ├── train_final_balanced.py
│   ├── train_finetune.py
│   ├── train_recall_focus.py
│   ├── predict.py
│   ├── gradcam_compare.py
│   └── gradcam_multiple.py
│
├── outputs/
│   ├── figures/
│   │   ├── confusion_matrix_finetune.png
│   │   ├── confusion_matrix_recall_focus.png
│   │   ├── training_curves_finetune.png
│   │   └── training_curves_recall_focus.png
│   │
│   └── reports/
│       ├── classification_report_finetune.txt
│       └── classification_report_recall_focus.txt
│
├── requirements.txt
├── .gitignore
└── README.md
Methods

The project includes the following components:

Balanced model training
Fine-tuning
Recall-focused training
X-ray image prediction
Grad-CAM visualization
Multiple Grad-CAM analysis
Model comparison
Confusion matrix analysis
Training curve analysis
Classification report generation
Model Training

Three main training approaches are included.

Final Balanced Training
python src/train_final_balanced.py

A balanced training strategy designed to improve overall classification performance.

Fine-Tuning
python src/train_finetune.py

A fine-tuning approach for improving the performance of the trained model.

Recall-Focused Training
python src/train_recall_focus.py

A training strategy focused on improving recall for abnormal X-ray detection.

Prediction

The predict.py script is used for making predictions on X-ray images.

python src/predict.py
Model Explainability

Grad-CAM is used to visualize the image regions that contribute to model predictions.

The repository includes:

gradcam_compare.py — comparison of Grad-CAM results
gradcam_multiple.py — Grad-CAM analysis for multiple images

These visualizations help make the model predictions more interpretable.

Results

The repository contains the generated evaluation results.

Confusion Matrices
confusion_matrix_finetune.png
confusion_matrix_recall_focus.png
Training Curves
training_curves_finetune.png
training_curves_recall_focus.png
Classification Reports
classification_report_finetune.txt
classification_report_recall_focus.txt

These files provide visual and numerical information about model performance.

Technologies
Python
PyTorch
Pandas
NumPy
Pillow
Matplotlib
OpenCV
Installation

Clone the repository and install the required dependencies:

pip install -r requirements.txt
Dataset

The project uses bone X-ray image data for binary abnormality classification.

Dataset files are not included in this repository.

Repository Notes

Large datasets, trained model weights, virtual environment files, and test images are excluded from version control using .gitignore.

The repository contains the source code, selected evaluation results, visualizations, and configuration files required to understand the project workflow.

Author

Hasan Barış

Biomedical Engineering Student