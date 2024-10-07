# Wine Quality Prediction

This project involves the prediction of wine quality using machine learning techniques based on various physicochemical properties of the wine. We explore the dataset through exploratory data analysis (EDA), build a machine learning model using Linear Regression, and evaluate the model's performance.

## Table of Contents

- [Installation](#installation)
- [Project Overview](#project-overview)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Model](#machine-learning-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run the code in this repository, you need to have Python installed. You can install the required packages by running:

```bash
pip install -r requirements.txt
```

### Required Libraries

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `os`

Make sure to have these libraries installed before running the code.

## Project Overview

The goal of this project is to predict the quality of red wine using a dataset that includes various chemical properties such as acidity, chlorides, pH, and more. The wine quality is categorized into three bins: low quality, medium quality, and high quality.

The dataset used is publicly available [here](https://raw.githubusercontent.com/amberkakkar01/Prediction-of-Wine-Quality/refs/heads/master/winequality-red.csv).

### Dataset

The dataset contains the following attributes:

- **fixed acidity**
- **volatile acidity**
- **citric acid**
- **residual sugar**
- **chlorides**
- **free sulfur dioxide**
- **total sulfur dioxide**
- **density**
- **pH**
- **sulphates**
- **alcohol**
- **quality** (Target variable)

### Binning

The wine quality is categorized into the following bins:

- Low Quality: 2-4
- Medium Quality: 4-6
- High Quality: 6-8

## Exploratory Data Analysis

Several plots are generated to visualize the distribution of the dataset and understand the correlation between various attributes. 

Plots include:

- Distribution of each attribute.
- Impact of elements such as volatile acidity, citric acid, pH, etc., on wine quality.
- A heatmap of correlations between features.
- Pairplot to visualize feature relationships with quality.

Each plot is saved dynamically in a folder called `plots/`.

## Machine Learning Model

The machine learning section focuses on training a linear regression model to predict the quality of wine.

Steps include:

1. **One Hot Encoding**: The quality bin is one-hot encoded for model training.
2. **Feature Scaling**: Features are standardized using `StandardScaler`.
3. **Train/Test Split**: The data is split into training and testing sets.
4. **Linear Regression**: A linear regression model is trained on the data.

### Model Evaluation

The performance of the model is evaluated using:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Median Absolute Error**
- **R² Score**

## Results

The linear regression model performs with a score of:

- **Mean Absolute Error (MAE)**: `0.37199817036183935`
- **Mean Squared Error (MSE)**: `0.18116217997072775`
- **Median Absolute Error**: `0.36234802099998076`
- **R² Score**: `0.7227841535437989`

The exact performance of the model may vary depending on the hyperparameters and data preprocessing techniques applied.