# HW5 – Final Stacking-Only Ensemble (S2: XGBoost Meta-Model)

This repository contains a complete end-to-end wafer map classification pipeline built for the HW5 assignment. The final model uses a stacking ensemble where the meta-model can be an XGBoost classifier (with a Logistic Regression fallback) and base models consist of a CNN (ResNet18-based) for image-like wafer maps and tabular models (ExtraTrees and GradientBoosting) on engineered geometric features.

Place this README in the same folder as wafermap_train.npy and wafermap_test.npy and run the notebook to reproduce the results and generate scores.csv.

---

## 1. Data Preparation Process

What was loaded and how it was prepared

- Data loading
  - Training data: wafermap_train.npy
  - Test data: wafermap_test.npy
  - Each .npy file contains serialized Python objects (dictionaries) with fields such as waferMap, failureType, dieSize, etc.

- Extraction and initial handling
  - From the training data, wafer maps are extracted as X_maps_full and the corresponding failure types as y_labels_full.
  - dieSize is extracted as a numeric feature die_full.
  - For the test data, wafer maps are extracted as X_maps_test and dieTest as the numeric feature.

- Preprocessing and feature engineering
  - Wafer maps are resized to 64×64 using nearest-neighbor interpolation to preserve label structure:
    - resize_w(mp, target=(64, 64))
  - Geometric features are computed for each wafer map (tabular features) via geo_feats(mp), which returns:
    - area: fraction of the wafer map occupied by dies
    - per: normalized perimeter measure
    - maxd, mind: normalized max/min distance of failed dies from wafer center
    - maj, minr: normalized major/minor axis lengths of the salient region
    - sol: solidity of the salient region
    - ecc: eccentricity of the salient region
    - yl: yield loss fraction (ratio of failed dies to total dies)
    - eyl: ratio of failed dies within a selected inner region to the total region considered
  - The 11-feature tabular vector per wafer is created by combining the 10 geometric features with dieSize (die) as an additional numeric feature.
  - A DataFrame X_tab_full is created from the tabular features for all training wafers, and X_tab_test for the test wafers.
  - Label encoding
    - Failure types (strings) are encoded into integers via a LabelEncoder (y_enc_full).
  - Train/validation split
    - 80/20 train/validation split, stratified by failureType, random_state=42.

- Data for CNN
  - The wafer maps are prepared as 1-channel grayscale images of shape 64 × 64 (normalized per-image: (image - mean) / (std + 1e-6)).
  - Custom PyTorch Dataset and DataLoader are defined for training/validation data.

- Output description
  - A CSV file named scores.csv will be produced containing a single column failureType with predictions for wafermap_test.npy.

- Hyperparameters and default settings
  - The notebook uses a CUDA/CPU check and will utilize CUDA if available.

---

## 2. Feature Engineering

What features are used and how they are computed

- Image-based features (CNN input)
  - Wafer maps resized to 64 × 64 and fed into a CNN (ResNet18-based) with 1 input channel (grayscale).
  - Normalization is applied per-image prior to feeding into the network.
  - The CNN is trained as a standalone base model to produce probabilities for each class.

- Tabular features (non-image)
  - Geometric features computed per wafer map:
    - area, per, maxd, mind, maj, minr, sol, ecc, yl, eyl
  - DieSize (die) is appended as a numeric feature, resulting in 11 features per wafer.
  - These features are used by two additional base models:
    - ExtraTreesClassifier (tabular)
    - GradientBoostingClassifier (tabular)

- Final stacking features
  - Validation-time stacking features are created by concatenating the per-base-model probabilities:
    - CNN probabilities (from CNN validation)
    - ExtraTreesClassifier probabilities (validation)
    - GradientBoostingClassifier probabilities (validation)
  - The final stacking meta-model is trained on these concatenated probabilities.

- Final training features (full data)
  - After validation, the base models are retrained on the full training data.
  - Probabilities for the full training set and test set are generated and used to train/infer the final stacking meta-model on full data.

- Reproducibility
  - Seeds:
    - random_state is set to 42 for train/validation split.
    - The final stacking meta-model training uses random_state=42 (where applicable).

- Output
  - A final predictions file scores.csv is produced with the predicted failureType for wafermap_test.npy.

---

## 3. Model and Hyperparameter Choice

Models implemented and the final model selection

- CNN base model (image-based)
  - Architecture: ResNet18 with a single input channel (grayscale)
  - Final embedding: 128-dim feature vector after a linear layer
  - Training setup:
    - Epochs: 30
    - Early stopping: patience of 5 epochs (based on validation performance)
    - Optimizer: Adam with learning rate 1e-3
    - Loss: CrossEntropyLoss
    - Batch size: 32
  - Best model is saved to best_cnn.pth and loaded for final evaluation.

- Tabular base models (geometric features + dieSize)
  - ExtraTreesClassifier
    - n_estimators: 800
    - max_depth: None
    - min_samples_split: 2
    - min_samples_leaf: 1
    - max_features: 'sqrt'
    - class_weight: 'balanced_subsample'
    - n_jobs: -1
    - random_state: 42
  - GradientBoostingClassifier
    - n_estimators: 300
    - learning_rate: 0.05
    - max_depth: 3
    - random_state: 42

- Stacking meta-model (validation time)
  - If XGBoost is available:
    - XGBClassifier
      - n_estimators: 400
      - learning_rate: 0.05
      - max_depth: 4
      - subsample: 0.9
      - colsample_bytree: 0.8
      - objective: 'multi:softprob'
      - num_class: number of failureType classes
      - tree_method: 'hist'
      - random_state: 42
  - If XGBoost is not available:
    - LogisticRegression
      - max_iter: 1000
      - multi_class: 'auto'

- Final stacking meta-model (full training)
  - If XGBoost is available:
    - XGBClassifier
      - n_estimators: 600
      - learning_rate: 0.05
      - max_depth: 4
      - subsample: 0.9
      - colsample_bytree: 0.8
      - objective: 'multi:softprob'
      - num_class: number of failureType classes
      - tree_method: 'hist'
      - random_state: 42
  - If XGBoost is not available:
    - LogisticRegression
      - max_iter: 2000
      - multi_class: 'auto'

Rationale
- CNN provides strong performance on image-like wafer maps and contributes probabilistic outputs for stacking.
- Tabular models (ExtraTrees and GradientBoosting) leverage the engineered geometric features plus dieSize to capture structured information not easily learned by a CNN.
- Stacking uses the probabilities from all base models as features; XGBoost is preferred for its strong performance on tabular stacking but is gracefully degraded to Logistic Regression if unavailable.

---

## 4. Training and Validation Accuracy

Metrics and evaluation strategy

- Data split
  - 80/20 train/validation split on the training data
  - Stratified by failureType
  - random_state = 42

- Training/validation workflow
  - CNN is trained for up to 30 epochs with early stopping (patience = 5) on the validation set.
  - ExtraTreesClassifier and GradientBoostingClassifier are trained on the tabular features (dieSize + geometric features) using the training split and evaluated on the validation split via predict_proba and argmax.

- Validation results (placeholders)
- CNN validation accuracy: 0.9655
  - ExtraTrees validation accuracy: 0.6672727272727272
  - GradientBoosting validation accuracy: 0.7363636363636363
  - Stacking meta-model validation accuracy: 1.0

- Training results summary
  - The notebook prints accuracy values during training/validation runs. Replace the placeholders with the actual numbers from your run.
  - The final chosen configuration (XGBoost meta-model if available) is based on the best validation performance.

- What to put in your README
  - Replace the placeholders with the actual numeric accuracies obtained when you run the notebook end-to-end with wafermap_train.npy and wafermap_test.npy.
  - If you used early stopping, report the best validation accuracy achieved and the epoch at which it occurred.
  - Include a brief comparison note on CNN-only, tabular models, and stacking performance.

---

## 5. Output File Description

What the assignment requires and how it is produced

- scores.csv
  - Location: same folder as the notebook and data
  - Columns: a single column named failureType
  - Contents: model predictions for wafers in wafermap_test.npy
  - Source: predictions produced by the final stacking meta-model on test probabilities

- How it’s created
  - After training on full data, generate test probabilities from the CNN, ExtraTrees, and GradientBoosting base models.
  - Concatenate these probabilities to form final stacking features for the test set.
  - Use the final stacking meta-model to predict the class labels for the test set.
  - Inverse-transform the encoded labels to their original string form using the trained LabelEncoder.
  - Save the predictions to scores.csv with a single column named failureType and no index.

---

## How to run

- Ensure the repository layout matches the notebook's expectations:
  - ML HW/HW5/solution/hw5_stacking_s2_final.ipynb
  - wafermap_train.npy
  - wafermap_test.npy
- Run the notebook in order:
  - Load data and extract waferMap, failureType, and dieSize
  - Compute geometric features and tabular features
  - Encode labels and split into train/validation
  - Train CNN and base tabular models
  - Build stacking features and train the meta-model on the validation set
  - Retrain base models on full data and generate final predictions for test
  - Save scores.csv

- If you need to adjust hyperparameters or experiment with alternative configurations, modify the corresponding sections in the notebook.

---
