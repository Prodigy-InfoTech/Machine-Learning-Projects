# Wine Quality Prediction Model

## About the Project 
I have Built a machine-learning model to predict the quality of wines by exploring their various chemical properties. Several factors other than age go into wine quality certification, which includes physiochemical tests like alcohol quantity, fixed acidity, volatile acidity, determination of density, pH, and more.

The following dataset has been used : 
https://raw.githubusercontent.com/amberkakkar01/Prediction-of-Wine-Quality/refs/heads/master/winequality-red.csv

## How I went about this Project
As the dataset is quite small, I decided to use an SVM(Support Vector Machine) model as the output was integers between 0 and 9. 
For the SVM I used the sklearn library and it's components. 
Also as the dataset was too small, I was getting an accurcay of only 60%.
So I augmented the dataset by simply doubling it by copy paste and setting the max iterations for the SVM fit to 250. 

This gave the model an accuracy of whopping **82.8%**!

Then I saved the model using The JobLib library