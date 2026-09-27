# WESAD Stress Classification — Machine Learning & Physiological Signal Analysis

A machine-learning project exploring the classification of physiological stress using wearable-sensor data from the **WESAD (Wearable Stress and Affect Detection)** dataset.

The project compares traditional machine-learning models and a 1D Convolutional Neural Network to investigate how physiological signals can be used to distinguish stress from non-stress states.

---

## Project Overview

Physiological stress can produce measurable changes in signals such as blood volume pulse, electrodermal activity, body temperature, and movement.

The goal of this project was to build and evaluate a machine-learning pipeline for stress classification using real-world physiological data.

The project covers:

* Physiological signal preprocessing
* Time-series windowing
* Statistical feature extraction
* Multiple machine-learning models
* Dimensionality reduction using PCA
* Hyperparameter tuning with GridSearchCV
* A 1D CNN using raw time-series sequences
* Model comparison using accuracy, precision, recall, and F1-score
* Feature-importance analysis

---

## Dataset

This project uses the **WESAD (Wearable Stress and Affect Detection)** dataset provided by the University of Siegen.

WESAD contains physiological and motion data collected from participants during different affective states, including stress.

### Signals Used

The project uses six signal channels:

* **BVP** — Blood Volume Pulse
* **EDA** — Electrodermal Activity
* **TEMP** — Temperature
* **ACC X** — Accelerometer X-axis
* **ACC Y** — Accelerometer Y-axis
* **ACC Z** — Accelerometer Z-axis

The experiment processed data from **15 participants** from the WESAD dataset.

The signals were divided into **60-second windows**, with the feature-based experiments using data sampled at **4 Hz**, resulting in 240 samples per channel per window.

The final feature dataset contained:

* **755 windows**
* **30 extracted features**
* **590 non-stress samples**
* **165 stress samples**

The original WESAD dataset is not included in this repository because of its size and distribution restrictions.

### Dataset Source

[WESAD — University of Siegen](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)

---

## Data Processing & Feature Engineering

Each 60-second window was processed to extract statistical characteristics from the six signal channels.

For each channel, the following features were calculated:

* Mean
* Standard deviation
* Minimum
* Maximum
* Median

This produced:

**6 signal channels × 5 statistical features = 30 features per window**

The WESAD labels were converted into a binary classification problem:

* **Stress**
* **No Stress**

Transition and undefined samples were excluded from the classification dataset.

---

## Models Tested

The project compares several approaches:

### Logistic Regression

Used as a baseline classification model.

### Random Forest

An ensemble tree-based classifier used to capture nonlinear relationships between the extracted physiological features.

### Multilayer Perceptron (MLP)

A feed-forward neural network trained on the engineered feature set.

### Logistic Regression + PCA

Principal Component Analysis was applied before Logistic Regression to investigate whether reducing the feature space could improve performance.

### Tuned Random Forest

GridSearchCV was used to search through different Random Forest hyperparameter combinations.

### 1D Convolutional Neural Network

A 1D CNN was trained directly on the raw six-channel time-series sequences rather than the engineered statistical features.

---

## Model Results

The models were evaluated using accuracy, precision, recall, and F1-score.

| Model                     |  Accuracy |  F1 Score |
| ------------------------- | --------: | --------: |
| Logistic Regression       |     87.4% |     0.689 |
| Random Forest             | **94.7%** | **0.867** |
| MLP                       |     94.0% |     0.857 |
| Logistic Regression + PCA |     84.1% |     0.600 |
| Tuned Random Forest       |     93.4% |     0.833 |
| 1D CNN                    |     84.8% |     0.582 |

### Best Observed Result

The **Random Forest** achieved the highest observed performance in this experiment:

* **Accuracy:** 94.7%
* **F1 Score:** 0.867

The MLP produced a similar result with 94.0% accuracy and an F1 score of 0.857.

---

## Key Findings

### 1. Random Forest performed best

Random Forest achieved the strongest results among the tested models.

This suggests that the statistical features extracted from the physiological signals contained useful information for distinguishing stress and non-stress samples in this experiment.

### 2. MLP performed similarly

The MLP achieved performance close to Random Forest, indicating that the engineered feature representation also worked well with a neural-network classifier.

### 3. PCA reduced performance

Applying PCA before Logistic Regression reduced performance compared with the original Logistic Regression model.

The original feature set therefore provided more useful information for this particular classification setup than the reduced feature representation.

### 4. The 1D CNN did not outperform the feature-based models

The CNN achieved lower performance than the best traditional machine-learning models.

This demonstrated that increasing model complexity does not automatically result in better performance. In this experiment, the engineered statistical features provided a more effective representation for the tested classifiers.

---

## Feature Importance

Feature-importance analysis was performed using the Random Forest model.

The analysis showed that **EDA standard deviation (`eda_std`) was the most important feature** among the extracted features.

This is particularly interesting because EDA is closely associated with changes in sympathetic nervous-system activity and can provide useful information when analyzing physiological responses to stress.

Other highly ranked features included measurements derived from BVP and additional EDA statistics.

The feature-importance analysis provides an additional perspective beyond overall model accuracy by showing which extracted characteristics contributed most strongly to the Random Forest's predictions.

---

## Evaluation Methodology

For the feature-based experiments, the extracted dataset was divided into training and testing sets using an **80/20 stratified random split**.

Feature scaling was performed using `StandardScaler`, with the scaler fitted on the training data before being applied to the test data.

PCA was fitted as part of the training workflow before transforming the corresponding data.

For the CNN experiment, the raw time-series sequences were also divided using an 80/20 split.

---

## Limitations

The evaluation methodology has an important limitation.

The dataset was split at the **window level rather than the participant level**.

Because multiple windows can originate from the same participant, windows from the same individual may appear in both the training and testing sets.

Therefore, the reported results should **not be interpreted as a strict measurement of how well the models generalize to completely unseen participants**.

A stronger evaluation for this type of physiological dataset would use a **subject-independent split**, where participants in the test set are completely excluded from the training data.

This is an important consideration when interpreting the reported 94.7% Random Forest accuracy.

### Potential Future Improvements

Future iterations could investigate:

* Subject-independent train/test splits
* Leave-one-subject-out cross-validation
* Additional physiological features
* More advanced time-series models
* Class-balancing techniques
* Larger-scale hyperparameter searches
* Additional validation across independent participant groups

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**
* **WESAD Dataset**

---

## Repository Structure

```text
MachineLearning/
│
├── ML_Proj_updated.ipynb
├── wesad_features.csv
├── wesad_model_metrics.csv
└── README.md
```

### Files

**`ML_Proj_updated.ipynb`**

The main notebook containing the data processing, feature extraction, model training, evaluation, visualizations, and analysis.

**`wesad_features.csv`**

The extracted feature dataset used by the feature-based machine-learning models.

**`wesad_model_metrics.csv`**

A summary of the model evaluation metrics.

---

## Running the Project

The WESAD dataset is not included in this repository.

To reproduce the experiment:

1. Download the WESAD dataset from the [University of Siegen](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html).
2. Open `ML_Proj_updated.ipynb` in a compatible Jupyter environment.
3. Provide the required WESAD dataset files.
4. Run the notebook cells in order.

The repository includes the extracted feature data and model metrics used in the project, allowing the results to be inspected without distributing the original WESAD dataset.

---

## What I Learned

This project provided practical experience with the end-to-end machine-learning workflow, from working with physiological time-series data to feature engineering, model development, evaluation, and interpretation.

Key areas of experience included:

* Processing real-world physiological sensor data
* Designing time-series windows
* Feature engineering
* Traditional machine-learning classification
* Neural-network-based classification
* Dimensionality reduction
* Hyperparameter optimization
* Model comparison
* Feature-importance analysis
* Interpreting model limitations

One of the main lessons from the project was that **more complex models do not necessarily produce better results**. In this experiment, Random Forest using engineered statistical features outperformed the 1D CNN trained on raw time-series data.

The project also highlighted the importance of choosing an evaluation strategy that matches the real-world question being investigated, particularly when working with data collected from multiple individuals.

The project also highlighted the importance of choosing an evaluation strategy that matches the real-world question being investigated, particularly when working with data collected from multiple individuals.
