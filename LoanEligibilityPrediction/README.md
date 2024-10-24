# Loan Eligibility Prediction

This project predicts loan eligibility based on various factors such as applicant income, loan amount, education, employment status, and more. The dataset contains loan information with categorical and numerical features, and this model processes the data and applies a Support Vector Machine (SVM) classifier to predict loan eligibility.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Prediction and Accuracy](#prediction-and-accuracy)
- [Best Model Parameters](#best-model-parameters)

## Introduction
This project aims to provide a solution for loan eligibility prediction. The main goal is to predict whether a loan will be approved (eligible) or not based on specific applicant and loan details. The data is preprocessed, scaled, and then used to train an SVM model. 

The code includes:
- Handling missing data using imputation
- Encoding categorical variables
- Feature scaling for numeric variables
- SVM model training and tuning for best parameters

## Dataset
The dataset includes the following columns:

- **Loan_ID**: Unique loan identifier
- **Gender**: Applicant's gender
- **Married**: Marital status of the applicant
- **Dependents**: Number of dependents
- **Education**: Education level (Graduate/Not Graduate)
- **Self_Employed**: Whether the applicant is self-employed
- **ApplicantIncome**: Applicant's income
- **CoapplicantIncome**: Co-applicant's income (if any)
- **LoanAmount**: Loan amount requested
- **Loan_Amount_Term**: Term of the loan in months
- **Credit_History**: Credit history status (1 for good, 0 for bad)
- **Property_Area**: Area where the property is located (Urban/Rural/Semiurban)
- **Loan_Status**: Loan approval status (Y = Approved, N = Not Approved)

### Link to the dataset:
https://www.kaggle.com/datasets/devzohaib/eligibility-prediction-for-loan
The dataset has also been uploaded in the directory. 

## Dependencies
This project requires the following Python libraries:
- `pandas`: For data manipulation and analysis
- `numpy`: For numerical operations
- `scikit-learn`: For machine learning models and preprocessing
- `matplotlib`: For data visualization (optional)

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```
## Preprocessing
Before training the model, several preprocessing steps are applied to the data:

- Missing Value Handling: Missing values in numeric columns are imputed with the mean, and in categorical columns with the most frequent value.
- One-Hot Encoding: Categorical variables such as 'Gender', 'Married', 'Self_Employed', 'Loan_Status', and 'Property_Area' are converted into numeric form using one-hot encoding.
- Scaling: All numeric features are scaled using StandardScaler to standardize the values, which improves the performance of the SVM model.

## Model Training
The model is trained using Support Vector Machines (SVM). The training process involves:

- Splitting the data into training and testing sets (80% training, 20% testing).
- Scaling the features before fitting the model.
- Training the SVM model with the default parameters and then tuning it with the best-found parameters (C=10, gamma=0.01, kernel='rbf').

## Prediction and Accuracy
The trained SVM model is used to make predictions on the test set. The accuracy of the predictions is calculated using the accuracy_score metric from scikit-learn.
![](https://github.com/Afreen-Kazi-1/Machine-Learning-Projects/blob/feature/LoanPrediction-Afreen/LoanEligibilityPrediction/Images/Screenshot%202024-10-22%20231303.png)

## How to Run
1. Clone the repository or download the project files.
2. Ensure that the required dependencies are installed (see the Dependencies section).
3. Place the dataset (CSV file) in the same directory as the code file.
4. Run the script modelpred.py using Python:
```bash
python modelpred.py
``` 
5.  View the printed accuracy and best model parameters in the terminal.
