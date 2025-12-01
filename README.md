# Customer Churn Prediction

This project uses the Telco Customer Churn dataset to build a simple machine learning model that predicts whether a customer will churn. The goal of the project is to demonstrate a complete workflow that includes data cleaning, feature encoding, model training and basic evaluation.

## Project Structure

````
Customer-Churn-Prediction/
│
├── data/
│      WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│
├── src/
│      model.py
│
└── README.md
````

## How to Run

1. Install required libraries:
   python -m pip install pandas scikit-learn matplotlib seaborn

2. Move into the src folder:
   cd src

3. Run the model:
   python model.py

The script will print a classification report and display a confusion matrix.

## Dataset

Telco Customer Churn dataset available on Kaggle.

## Notes

The model used here is a Random Forest classifier. You can experiment with other models, additional preprocessing or hyperparameter tuning to extend the project.
