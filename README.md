# Customer Churn Prediction Application

This is a Streamlit-based web application that predicts customer churn using machine learning. The application uses a synthetic dataset and Logistic Regression to make predictions.

## Features

- Interactive user interface for inputting customer details
- Real-time churn prediction
- Visual data analysis
- Feature importance visualization
- Dataset statistics

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

To run the application, use the following command:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## How to Use

1. Use the sliders in the sidebar to input customer details:
   - Monthly Usage (minutes)
   - Contract Length (months)
   - Customer Age (years)
   - Support Calls (last month)
   - Monthly Spend ($)

2. The application will automatically:
   - Calculate the churn probability
   - Show the prediction (Will Churn/Will Not Churn)
   - Display visualizations of the data
   - Show feature importance
   - Display dataset statistics

## Model Details

The application uses:
- Logistic Regression for prediction
- StandardScaler for feature scaling
- Synthetic data generation for demonstration
- 80/20 train-test split 