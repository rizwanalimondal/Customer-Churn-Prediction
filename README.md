# Customer Churn Prediction

This project uses the Telco Customer Churn dataset to build a simple model that predicts whether a customer is likely to discontinue their service. The aim is to demonstrate a practical end-to-end approach: loading the data, preparing it, encoding categorical fields, training a baseline model and evaluating its performance.

The dataset contains information about customer demographics, account details, contract type, services used and billing patterns. The target column is "Churn", which indicates whether the customer left the service.

---

## Project Structure
````
Customer-Churn-Prediction/
│
├── data/
│      WA_Fn-UseC_-Telco-Customer-Churn.csv  
│
├── notebooks/  
│      (optional area for Jupyter notebooks)  
│
├── src/
│      model.py  
│
└── README.md
````
---

## Data Description (Key Columns)

Below is a brief description of the main fields in the dataset:

- **gender** – Male or female  
- **SeniorCitizen** – Whether the customer is a senior citizen  
- **Partner** – Whether the customer has a partner  
- **Dependents** – Whether the customer has dependents  
- **tenure** – Number of months the customer has stayed with the company  
- **PhoneService / InternetService** – Types of services subscribed  
- **Contract** – Month-to-month, one-year or two-year contract  
- **PaperlessBilling** – Whether billing is paperless  
- **PaymentMethod** – Billing method used  
- **MonthlyCharges** – Current monthly bill  
- **TotalCharges** – Total amount billed  
- **Churn** – Target variable (Yes = customer left)

This dataset comes from the Telco Customer Churn data published on Kaggle.

---

## Approach

1. Load the dataset and handle missing values.  
2. Convert the TotalCharges column to numeric format.  
3. Encode all categorical features using label encoding.  
4. Split the dataset into training and testing sets.  
5. Train a Random Forest classifier.  
6. Evaluate the model using accuracy, precision, recall and f1-score.  
7. Plot a confusion matrix to visualize classification performance.

This gives a straightforward baseline model. The project can be extended later using better preprocessing, model selection or hyperparameter tuning.

---

## Model Performance Summary

(These values may vary slightly depending on randomness.)

- **Accuracy:** ~78%  
- **Precision and Recall:** Higher for non-churn class, lower for churn class  
- **Confusion Matrix:** Shows that the model predicts non-churn cases more accurately than churn cases

The exact classification report and confusion matrix are generated when running the script.

---

## How to Run

1. Install the required libraries:

python -m pip install pandas scikit-learn matplotlib seaborn

2. Move into the project folder (if not already there):

cd Customer-Churn-Prediction

3. Run the model from the src directory:

cd src
python model.py


This will print the classification report in the terminal and display the confusion matrix.

---

## Future Improvements

- Add exploratory data analysis (EDA)  
- Build the project in a Jupyter notebook for step-by-step explanation  
- Try additional models such as Logistic Regression, XGBoost or Gradient Boosting  
- Perform hyperparameter tuning  
- Add more detailed performance metrics  
- Create a Streamlit dashboard for interactive churn exploration

---

## Notes

This project is intended as a simple baseline example for churn prediction and can be expanded based on use-case requirements.

