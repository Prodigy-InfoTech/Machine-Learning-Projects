#Heart Disease Detection Project
This project aims to predict the likelihood of heart disease in individuals based on various medical attributes. The model is built using Logistic Regression, a well-known machine learning algorithm for binary classification, and deployed as an interactive web application using Streamlit.

##Features
1.User Input: The application accepts user input for key health metrics such as age, sex, cholesterol levels, blood pressure, and other medical parameters.
2.Real-time Predictions: After entering the data, the model provides instant predictions on whether the individual is at risk of heart disease.
3.Visualizations: The app includes helpful visualizations for better understanding of the data and prediction results.
4.Easy-to-Use Interface: Powered by Streamlit, the interface is intuitive, allowing users to quickly input their data and receive feedback.
##Technologies Used
Machine Learning: Logistic Regression
Web Framework: Streamlit
Programming Language: Python
##Dataset
The model is trained on a public dataset containing heart disease records. Each record includes various medical factors that contribute to the likelihood of heart disease.

##How to Run
Clone the repository:

"**git clone https://github.com/yourusername/heart-disease-detection.git**"
Navigate to the project directory:

"**cd heart-disease-detection**"
Install the required dependencies:

"**pip install -r requirements.txt**"
Run the Streamlit app:

"**streamlit run app.py**"
##Usage
Open the Streamlit app in your browser.
Enter the required health metrics in the input form.
Click "Predict" to get the prediction result on heart disease risk.
##Model
The Logistic Regression model was trained using health data, with features such as age, sex, resting blood pressure, cholesterol, maximum heart rate, and more. Logistic Regression was chosen for its simplicity and efficiency in binary classification tasks.
