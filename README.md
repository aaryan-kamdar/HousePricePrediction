# House Price Prediction

## Overview

This project builds an end-to-end machine learning pipeline to predict residential house prices using the Ames Housing Dataset. The goal is to simulate a real-world use case where a real estate company can automatically suggest a listing price based on a property's characteristics.

The project covers the complete supervised learning workflow:

* Data cleaning
* Handling missing values
* Feature engineering
* Categorical encoding
* Feature scaling
* Model training and evaluation
* Hyperparameter tuning with GridSearchCV
* Model serialization using Pickle
* Command-line inference using `argparse`

---

## Problem Statement

A real estate startup wants to automatically estimate the sale price of a property as soon as a seller enters its details.

Given information such as:

* Lot Area
* Neighborhood
* Overall Quality
* Basement Quality
* Living Area
* Number of Bedrooms
* Garage Size
* Year Built
* Number of Bathrooms

the system predicts the expected sale price of the house.

This project demonstrates how machine learning models can be deployed as reusable tools that other developers or applications can interact with.

---

## Dataset

This project uses the Ames Housing Dataset.

The dataset contains information about residential homes in Ames, Iowa, including structural characteristics, location, and sale prices.

### Features Used

| Feature                      |
| ---------------------------- |
| Lot Area                     |
| Neighborhood                 |
| Overall Qual                 |
| Overall Cond                 |
| Year Built                   |
| Year Remod/Add               |
| Bsmt Qual                    |
| Gr Liv Area                  |
| Full Bath                    |
| Bedroom AbvGr                |
| TotRms AbvGrd                |
| Garage Cars                  |
| Garage Area                  |
| TotalSF (engineered feature) |

---

## Machine Learning Workflow

### Data Preprocessing

* Filled missing numerical values using the median.
* Filled missing basement quality values with `"None"`.
* Applied `log1p()` transformation to skewed numerical features.
* One-hot encoded the `Neighborhood` feature.
* Ordinal encoded `Bsmt Qual`.
* Created an engineered feature:

```python
TotalSF = Total Bsmt SF + 1st Flr SF + 2nd Flr SF
```

* Standardized features using `StandardScaler`.

---

## Models Trained

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

The final model selected for deployment was a tuned XGBoost model.

---

## Hyperparameter Tuning

Hyperparameter optimization was performed using `GridSearchCV` with 5-fold cross-validation.

### Parameters Tuned

* Number of estimators
* Learning rate
* Maximum tree depth
* Subsample ratio
* Column sample ratio

---

## Project Structure

```text
house-price-prediction/
│
├── train.ipynb
├── predict.py
├── AmesHousing-selected-columns.csv
├── xg_boost_best.sav
├── scaler.sav
├── bsmt_qual_encoder.sav
├── columns.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/house-price-prediction.git

cd house-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Predictions

The project includes a command-line prediction tool built using `argparse`.

### Example

```bash
python predict.py \
--lot_area 10000 \
--overall_qual 7 \
--overall_cond 6 \
--year_built 2000 \
--year_remod_add 2010 \
--bsmt_qual Gd \
--gr_liv_area 1800 \
--full_bath 2 \
--bedroom_abvgr 3 \
--totrms_abvgrd 7 \
--garage_cars 2 \
--garage_area 450 \
--neighborhood NAmes \
--total_bsmt_sf 900 \
--first_flr_sf 1200 \
--second_flr_sf 600
```

### Sample Output

```text
Predicted House Price: $204,007.48
```

---

## Saved Artifacts

The following files are generated after training:

| File                    | Description                                |
| ----------------------- | ------------------------------------------ |
| `xg_boost_best.sav`     | Trained XGBoost model                      |
| `scaler.sav`            | StandardScaler used during training        |
| `bsmt_qual_encoder.sav` | Ordinal encoder for Basement Quality       |
| `columns.pkl`           | Stores training column order for inference |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Pickle
* Argparse

---

## Key Learnings

Through this project, I learned:

* Data preprocessing techniques.
* Handling missing values.
* Feature engineering.
* Regression modeling.
* Model evaluation using RMSE and MAE.
* Hyperparameter tuning with GridSearchCV.
* Saving and loading machine learning models.
* Building command-line tools with `argparse`.
* Reproducing preprocessing during inference.
* Structuring a complete machine learning project for deployment.

---


