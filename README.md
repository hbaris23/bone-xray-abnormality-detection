# Bone X-Ray Abnormality Detection

A deep learning-based computer vision project for detecting abnormalities in bone X-ray images.

## Overview

This project focuses on the binary classification of bone X-ray images using deep learning techniques.

Several training strategies were developed and evaluated to improve the detection of abnormal X-ray images. The project also includes Grad-CAM-based visual analysis to provide an interpretable view of the regions that influence model predictions.

The system is designed as a research and educational prototype for medical image analysis and should not be considered a clinical diagnostic tool.

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
Multiple-image Grad-CAM analysis
Model comparison
Confusion matrix analysis
Training curve analysis
Classification report generation
Model Training

Three main training strategies are included.

1. Final Balanced Training
python src/train_final_balanced.py

A balanced training strategy designed to improve the overall classification performance of the model.

2. Fine-Tuning
python src/train_finetune.py

A fine-tuning approach used to further improve model performance.

3. Recall-Focused Training
python src/train_recall_focus.py

A training strategy focused on improving recall for abnormal X-ray detection.

Prediction

The predict.py script is used to perform predictions on bone X-ray images.

python src/predict.py

The prediction pipeline allows the trained model to classify input X-ray images into the defined categories.

Model Explainability

Grad-CAM is used to visualize the image regions that contribute to model predictions.

The repository includes:

gradcam_compare.py — comparison of Grad-CAM results
gradcam_multiple.py — Grad-CAM analysis for multiple images

These visualizations provide an interpretable representation of the model's decision-making process and help investigate which regions of an X-ray image influence the prediction.

Results

The repository contains selected evaluation results generated during model development.

Confusion Matrices
confusion_matrix_finetune.png
confusion_matrix_recall_focus.png
Training Curves
training_curves_finetune.png
training_curves_recall_focus.png
Classification Reports
classification_report_finetune.txt
classification_report_recall_focus.txt

These files provide visual and numerical information about model performance, including classification behavior and training progress.

Technologies
Python
PyTorch
Pandas
NumPy
Pillow
Matplotlib
OpenCV
Installation

Clone the repository:

git clone https://github.com/hbaris23/bone-xray-abnormality-detection.git
cd bone-xray-abnormality-detection

Install the required dependencies:

pip install -r requirements.txt
Dataset

The project uses bone X-ray image data for binary abnormality classification.

The dataset is not included in this repository due to size and data management considerations.

Repository Notes

Large datasets, trained model weights, virtual environment files, and test images are excluded from version control using .gitignore.

The repository contains the source code, selected evaluation results, visualizations, and configuration files required to understand the project workflow.

Future Improvements

Possible future improvements include:

Evaluation on larger and more diverse datasets
Additional data augmentation strategies
Comparison with different deep learning architectures
Improved abnormality localization
Additional explainability methods
More comprehensive model evaluation
Author

Hasan Barış

Biomedical Engineering Student
