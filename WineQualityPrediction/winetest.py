import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #for scaling
from sklearn.model_selection import train_test_split # for splitting the dataset
from sklearn.svm import SVC # for svc training
from sklearn.metrics import accuracy_score # function to calculate accuracy
from sklearn.impute import SimpleImputer # for handling missing values
import joblib #for saving the model after training. 

data = pd.read_csv('C:/Users/Afreen/.vscode/Coding/OpenCV/tasks/Hacktoberfest/winequality-red.csv')


# Handling missing data
X = data.drop(columns=['quality'])
y = data['quality']
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.1, random_state=42)

Scaler = StandardScaler() #scaling to standardize the values
X_train_scaled = Scaler.fit_transform(X_train)
X_test_scaled = Scaler.transform(X_test)

# Train the SVM model
svm = SVC(max_iter=250)
svm.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred) # Calculating the accuracy score
print(f'Accuracy: {accuracy}')

# Best parameters
best_params = {
    'C' : 1000,
    'gamma' : 0.01 ,
    'kernel' : 'rbf'
}
print(f'Best Parameters: {best_params}')


# Train the best model
best_svm = SVC(C=best_params['C'], gamma=best_params['gamma'] ,kernel = best_params['kernel'])
best_svm.fit(X_train, y_train)
#accuracy for best model

y_pred_best = best_svm.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f'Best Accuracy: {best_accuracy}')

# Save the trained model and the scaler
joblib.dump(svm, 'svm_model.pkl')  # Save the trained model
joblib.dump(Scaler, 'scaler.pkl')  # Save the scaler for standardizing user input

print("Model and Scaler saved successfully!")

# Load the model and scaler
loaded_svm = joblib.load('svm_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# ---- TEST THE MODEL WITH RANDOM USER INPUT ----
# Assume the user provides input with the same number of features as the dataset
# Example: Replace the list below with a random user input for wine quality features
user_input = np.array([[6.0,0.31,0.47,3.6,0.067,18.0,42.0,0.99549,3.39,0.66,11.0]])  # Example input

# Standardize the user input
user_input_scaled = loaded_scaler.transform(user_input)

# Predict using the loaded model
user_pred = loaded_svm.predict(user_input_scaled)
print(f"Predicted quality for the input data: {user_pred[0]}")