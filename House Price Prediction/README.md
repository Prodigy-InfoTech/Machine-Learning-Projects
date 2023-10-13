
# House Price Prediction Project
This GitHub repository contains a Jupyter Notebook (house-price-prediction.ipynb) that demonstrates a comprehensive analysis and prediction model for house prices using the Boston Housing Dataset. The project is written in Python and relies on various libraries such as NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, and XGBoost.

## Overview

The project's primary objectives are as follows:

1.  Data collection and preprocessing.
2.  Data analysis and visualization to gain insights into the dataset.
3.  Building and training multiple regression models to predict house prices.
4.  Model evaluation and feature importance analysis.

## Getting Started

Follow the steps below to set up and run the project on your local machine.

### Prerequisites

-   [Jupyter Notebook](https://jupyter.org/)
-   Python 3.6 or later
-   Required Python packages can be installed using the provided `requirements.txt` file.

bashCopy code

`pip install -r requirements.txt` 

### Installation

1.  Clone this repository to your local machine:

bashCopy code

`git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction` 

2.  Launch Jupyter Notebook and open the `house-price-prediction.ipynb` file to run the project.

## Dataset

The project uses the [Boston Housing Dataset](https://www.kaggle.com/datasets/altavish/boston-housing-dataset) obtained from Kaggle. This dataset contains 14 features, including factors like crime rate, room count, and pupil-teacher ratio, used to predict the median value of owner-occupied homes (MEDV).

## Notebook

The project is structured in the following sections:

-   Data Collection: Fetches the dataset from Kaggle and loads it into a Pandas DataFrame.
-   Data Preprocessing: Handles missing data and standardizes some of the features.
-   Data Analysis: Provides insights into the dataset through visualizations and correlation analysis.
-   Model Building and Evaluation: Trains various regression models, including Linear Regression, Decision Tree Regression, Random Forest Regression, Extra Trees Regression, and XGBoost Regression as well as evaluates the performance of each model using Mean Squared Error and cross-validation.

## Model Performance

The project builds and evaluates multiple regression models, and you can review their performance metrics in the Jupyter Notebook. Additionally, the feature importance analysis provides insights into which factors have the most influence on house price predictions.
