import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv("C:/Users/Afreen/.vscode/Coding/OpenCV/tasks/Hacktoberfest/loaneligibility/Loan_Data.csv")

# Encode 'Education' (Graduate = 1, Not Graduate = 0)
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})

# One-hot encode 'Property_Area' (Rural, Urban, Semiurban)
data = pd.get_dummies(data, columns=['Property_Area'], drop_first=True)

# Convert '3+' in 'Dependents' to 3 and fill missing values with 0
data['Dependents'] = data['Dependents'].replace('3+', 3)
data['Dependents'] = data['Dependents'].fillna(0).astype(int)

# One-hot encode categorical columns including Loan_Status
data = pd.get_dummies(data, columns=['Gender', 'Married', 'Self_Employed', 'Loan_Status'], drop_first=True)

# Define numeric and categorical columns
num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'ApplicantIncome', 'CoapplicantIncome']
cat_cols = ['Gender_Male', 'Married_Yes', 'Dependents', 'Self_Employed_Yes']

# Impute missing values
imputer_num = SimpleImputer(strategy='mean')
data[num_cols] = imputer_num.fit_transform(data[num_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')
data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])

# Scaling numeric columns
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Drop Loan_ID (identifier column)
data = data.drop(columns=['Loan_ID'])

# Separate the features and target
X = data.drop(columns=['Loan_Status_Y'])  # Adjust this according to the one-hot encoding
y = data['Loan_Status_Y']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82)

# Scale the features
Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train)
X_test_scaled = Scaler.transform(X_test)

# Train the SVM model
svm = SVC()
svm.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Define best parameters
best_params = {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
print(f'Best Parameters: {best_params}')

# Train the best model
best_svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
best_svm.fit(X_train_scaled, y_train)

# Accuracy for the best model
y_pred_best = best_svm.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f'Best Accuracy: {best_accuracy}')
