### Dataset
This project uses the WESAD (Wearable Stress and Affect Detection) dataset provided by the University of Siegen.

Dataset Download:
The dataset used in this project (WESAD) can be downloaded from the 
University of Siegen’s official hosting page:
https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

The dataset is not included in this repository due to size and licensing constraints. 
Please download it manually from the links above before running the notebook.


This project develops a machine-learning pipeline for detecting physiological stress using data from the WESAD dataset. The work includes:

Full preprocessing of wrist-based physiological signals

Extraction of statistical features from BVP, EDA, TEMP, and ACC

Training multiple machine-learning models

Applying PCA for dimensionality reduction

Hyperparameter tuning using GridSearchCV

Implementing a 1D CNN on raw time-series sequences

Comparing all models using accuracy, precision, recall, and F1-score

This repository contains all code and results, excluding the dataset itself.
