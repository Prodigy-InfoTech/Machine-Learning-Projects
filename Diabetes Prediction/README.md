## DIABETES PREDICTION

**GOAL**

The goal is to predict if a person is having Diabetes or not using the diabetes dataset. This dataset contains features as Pregnancies, Glucose, BloodPressure, SkinThickness, BMI, Insulin, DiabetesPedegreeFunction and Age. On the basis of this feature, we have to predict the outcome.

Dataset can be downloaded from [here](https://www.kaggle.com/mathchi/diabetes-data-set)


**WHAT I HAD DONE**

- Using libraries and pre-processing techniques, I had replaced that 0 with median of that particular feature.
- Then used Correlation coefficients to measure how strong a relationship is between two variables and to detect the multicollinearity.
- With nothing to be found, I applied Standardization to convert those in such a way that thier mean=0 and Std.Deviation = 1.
- Split the datsaset into for parts : Training Data and its respective outcome, Test Data and its respective outcome.
- Then implement classification models : Logistic Regression, Decision Trees, SVM and Random Forest on the training data.
- Finally calculate accuracy of every model on the training and test data along with reviewing thier respective confusion matrix.
- After training and predicting values, I had saved the respective model using pickle and os library.


**MODELS USED**

-  Logistic Regression
-  Decision Tree Classifier 
-  Support Vector Machine
-  Random Forest Classifier


**RESULTS**

By using Logistic Regression I got 
 ```
    Accuracy of training data: 0.7687296416938111
    Accuracy of testing data: 0.8181818181818182
 ``` 
 
By using Decision Tree Classifier I got 
 ```
    Accuracy of training data: 1.00
    Accuracy of testing data: 0.7597402597402597
 ``` 
By using SVM I got 
 ```
    Accuracy of training data: 0.8143322475570033
    Accuracy of testing data: 0.7727272727272727
 ``` 

By using Random Forest Classifier I got 
 ```
    Accuracy of training data: 1.00
    Accuracy of testing data:  0.8181818181818182
 ``` 
 
Comparing all those scores scored by the machine learning algorithms, it is clear that Random Forest Classifier and Logistic Regression Algorithm have got a higher note and after this, we can consider SVM algorithm which is having relatively good score as compared to the Decision Tree Classifier.
 
The models are deployed successfully! Good to go.
 
