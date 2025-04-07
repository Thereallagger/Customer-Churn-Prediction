import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Customer Churn Prediction Tool")
st.markdown("This app helps you predict which customers are likely to churn based on their usage patterns and characteristics.")

# Sidebar with navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Data Exploration", "Model Training", "Predictions"])

# Store feature information in session state
if 'feature_info' not in st.session_state:
    st.session_state['feature_info'] = {
        'categorical_cols': [],
        'numerical_cols': []
    }

# Function to convert churn values to binary
def convert_churn_to_binary(df):
    if 'Churn' in df.columns and df['Churn'].dtype == object:
        churn_map = {
            'Yes': 1, 'No': 0,
            'Y': 1, 'N': 0,
            'True': 1, 'False': 0,
            '1': 1, '0': 0,
            1: 1, 0: 0,
            True: 1, False: 0
        }
        df['Churn'] = df['Churn'].map(lambda x: churn_map.get(x, x))
    return df

# Function to verify required columns
def verify_columns(df):
    required_columns = [
        'MonthlyUsage', 'ContractLength', 
        'CustomerServiceCalls', 'TenureMonths', 'MonthlyCharges', 
        'Churn'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.info(f"Available columns: {df.columns.tolist()}")
        return False
    return True

# Function for data exploration
def explore_data(df):
    st.header("Data Exploration")
    
    # Display dataset info
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Info")
        st.write(f"Total records: {df.shape[0]}")
        st.write(f"Total features: {df.shape[1]}")
    
    with col2:
        st.subheader("Churn Distribution")
        churn_count = df['Churn'].value_counts()
        churn_percent = df['Churn'].value_counts(normalize=True) * 100
        
        # Display churn distribution as a pie chart
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(churn_count, labels=['Not Churned', 'Churned'] if churn_count.index[0] == 0 else ['Churned', 'Not Churned'], 
              autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    # Data sample
    st.subheader("Data Sample")
    st.dataframe(df.head())
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    
    # Check for missing values
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.dataframe(missing[missing > 0])
    else:
        st.info("No missing values found.")
    
    # Visualizations
    st.subheader("Visualizations")
    
    # Visualization selection
    viz_option = st.selectbox("Select Visualization", [
        "Churn by Monthly Usage", 
        "Churn by Contract Length", 
        "Correlation Heatmap"
    ])
    
    if viz_option == "Churn by Monthly Usage":
        if 'MonthlyUsage' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='MonthlyUsage', hue='Churn', data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("MonthlyUsage column not found in dataset.")
    
    elif viz_option == "Churn by Contract Length":
        if 'ContractLength' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='ContractLength', hue='Churn', data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("ContractLength column not found in dataset.")
    
    elif viz_option == "Correlation Heatmap":
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for correlation analysis.")

# Function to build and train the model
def build_model(df):
    st.header("Model Training")
    
    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Store column information in session state for prediction
    st.session_state['feature_info']['categorical_cols'] = categorical_cols
    st.session_state['feature_info']['numerical_cols'] = numerical_cols
    
    st.subheader("Feature Information")
    st.write(f"Categorical features: {categorical_cols}")
    st.write(f"Numerical features: {numerical_cols}")
    
    # Model parameters
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 50, 30) / 100
        random_state = st.number_input("Random State", 0, 100, 42)
    
    with col2:
        max_iter = st.number_input("Max Iterations", 100, 5000, 1000)
    
    # Train button
    if st.button("Train Model"):
        with st.spinner("Training model... This may take a moment."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Create preprocessing steps
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            numerical_transformer = StandardScaler()
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ], remainder='passthrough'
            )
            
            # Create model pipeline
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(random_state=random_state, max_iter=max_iter))
            ])
            
            # Train model
            model.fit(X_train, y_train)
            
            # Save model and its data requirements to session state
            st.session_state['model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
            # Also store the expected columns for predictions
            st.session_state['expected_columns'] = X.columns.tolist()
            
            st.success("Model trained successfully!")
            
            # Save model to disk
            if st.button("Save Model to Disk"):
                directory = 'models'
                os.makedirs(directory, exist_ok=True)
                filepath = os.path.join(directory, 'churn_prediction_model.pkl')
                
                # Save model and feature info together
                model_data = {
                    'model': model,
                    'expected_columns': X.columns.tolist(),
                    'categorical_cols': categorical_cols,
                    'numerical_cols': numerical_cols
                }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                
                st.success(f"Model saved to {filepath}")
    
    # If model exists in session state, show evaluation
    if 'model' in st.session_state:
        st.header("Model Evaluation")
        
        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Churn', 'Churn'],
                        yticklabels=['Not Churn', 'Churn'], ax=ax)
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)
        
        with col2:
            st.subheader("ROC Curve")
            auc_score = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            st.pyplot(fig)
        
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Feature importance
        if hasattr(model['classifier'], 'coef_'):
            st.subheader("Feature Importance")
            
            try:
                # Get feature names after preprocessing
                categorical_cols = st.session_state['feature_info']['categorical_cols']
                numerical_cols = st.session_state['feature_info']['numerical_cols']
                
                if categorical_cols:
                    cat_features = model['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols)
                    feature_names = np.concatenate([numerical_cols, cat_features])
                else:
                    feature_names = numerical_cols
                
                # Get coefficients
                coefficients = model['classifier'].coef_[0]
                
                # Create DataFrame
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': coefficients
                })
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                
                # Display as bar chart
                fig, ax = plt.subplots(figsize=(10, 8))
                top_n = min(15, len(feature_importance))  # Show top 15 features or less
                top_features = feature_importance.head(top_n)
                sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Could not analyze feature importance: {str(e)}")

# Function to make predictions
def make_predictions():
    st.header("Churn Predictions")
    
    # Check if model exists
    if 'model' not in st.session_state:
        st.warning("Please train a model first or upload a pre-trained model.")
        
        # Option to upload pre-trained model
        uploaded_model = st.file_uploader("Upload a pre-trained model (.pkl)", type=["pkl"])
        if uploaded_model is not None:
            try:
                # Load model and related information
                model_data = pickle.load(uploaded_model)
                
                # Check if model data is in new format with feature info
                if isinstance(model_data, dict) and 'model' in model_data:
                    st.session_state['model'] = model_data['model']
                    st.session_state['expected_columns'] = model_data['expected_columns']
                    st.session_state['feature_info']['categorical_cols'] = model_data['categorical_cols']
                    st.session_state['feature_info']['numerical_cols'] = model_data['numerical_cols']
                else:
                    # Legacy format - just the model
                    st.session_state['model'] = model_data
                    st.warning("Model loaded, but feature information is missing. Predictions may fail.")
                    
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
            
    # If model exists, show prediction options
    if 'model' in st.session_state:
        st.subheader("Make Predictions")
        
        # Options for prediction
        pred_option = st.radio("Select prediction method", ["Upload Data", "Enter Single Customer"])
        
        if pred_option == "Upload Data":
            uploaded_file = st.file_uploader("Upload customer data", type=["csv", "xlsx"])
            
            if uploaded_file is not None:
                try:
                    # Load data
                    if uploaded_file.name.lower().endswith('.csv'):
                        new_data = pd.read_csv(uploaded_file)
                    else:
                        new_data = pd.read_excel(uploaded_file)
                    
                    # Display data
                    st.subheader("Uploaded Data Preview")
                    st.dataframe(new_data.head())
                    
                    # Check for Churn column and remove if exists
                    if 'Churn' in new_data.columns:
                        new_data = new_data.drop('Churn', axis=1)
                    
                    # Make predictions
                    if st.button("Generate Predictions"):
                        model = st.session_state['model']
                        
                        # Check column compatibility
                        if 'expected_columns' in st.session_state:
                            expected_cols = st.session_state['expected_columns']
                            missing_cols = [col for col in expected_cols if col not in new_data.columns]
                            extra_cols = [col for col in new_data.columns if col not in expected_cols]
                            
                            if missing_cols:
                                st.error(f"Missing columns required by the model: {missing_cols}")
                                return
                            
                            if extra_cols:
                                st.warning(f"Extra columns found that will be ignored: {extra_cols}")
                                new_data = new_data[expected_cols]
                        
                        # Predict
                        try:
                            churn_pred = model.predict(new_data)
                            churn_prob = model.predict_proba(new_data)[:, 1]
                            
                            # Add predictions to data
                            results = new_data.copy()
                            results['Churn_Prediction'] = churn_pred
                            results['Churn_Probability'] = churn_prob
                            results['Churn_Result'] = results['Churn_Prediction'].map({1: 'Yes', 0: 'No'})
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(results)
                            
                            # Download results
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions",
                                data=csv,
                                file_name="churn_predictions.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
                            st.info("Make sure the columns in your data match what the model was trained on.")
                
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
        
        elif pred_option == "Enter Single Customer":
            st.subheader("Enter Customer Information")
            
            # Get expected columns from session state if available
            expected_cols = st.session_state.get('expected_columns', [
                'MonthlyUsage', 'ContractLength', 'CustomerServiceCalls', 'TenureMonths', 'MonthlyCharges'
            ])
            
            # Get categorical columns from feature info
            categorical_cols = st.session_state['feature_info'].get('categorical_cols', [
                'MonthlyUsage', 'ContractLength'
            ])
            
            # Create columns in UI
            col1, col2 = st.columns(2)
            
            # Initialize dictionary to store customer data
            customer_data = {}
            
            # Add fields based on expected columns
            with col1:
                if 'MonthlyUsage' in expected_cols:
                    # Get unique values from the dataset if available
                    if 'data' in st.session_state and st.session_state['data'] is not None and 'MonthlyUsage' in st.session_state['data']:
                        usage_options = sorted(st.session_state['data']['MonthlyUsage'].unique().tolist())
                    else:
                        usage_options = ["Low", "Medium", "High"]
                    customer_data['MonthlyUsage'] = st.selectbox("Monthly Usage", usage_options)
                
                if 'ContractLength' in expected_cols:
                    # Get unique values from the dataset if available
                    if 'data' in st.session_state and st.session_state['data'] is not None and 'ContractLength' in st.session_state['data']:
                        contract_options = sorted(st.session_state['data']['ContractLength'].unique().tolist())
                    else:
                        contract_options = ["Monthly", "One year", "Two year"]
                    customer_data['ContractLength'] = st.selectbox("Contract Length", contract_options)
                
                if 'CustomerServiceCalls' in expected_cols:
                    customer_data['CustomerServiceCalls'] = st.number_input("Customer Service Calls", min_value=0, step=1)
            
            with col2:
                if 'TenureMonths' in expected_cols:
                    customer_data['TenureMonths'] = st.number_input("Tenure in Months", min_value=0, step=1)
                
                if 'MonthlyCharges' in expected_cols:
                    customer_data['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0, step=0.01)
                
                # Handle any other columns that might be in the dataset
                extra_cols = [col for col in expected_cols if col not in customer_data and col not in ['Churn']]
                for col in extra_cols:
                    if col in categorical_cols:
                        customer_data[col] = st.selectbox(f"{col}", ["Value1", "Value2", "Value3"])
                    else:
                        customer_data[col] = st.number_input(f"{col}", step=0.01)
            
            # Make prediction
            if st.button("Predict Churn"):
                # Create DataFrame for the new customer
                new_customer = pd.DataFrame({col: [val] for col, val in customer_data.items()})
                
                # Ensure columns match what's expected
                if 'expected_columns' in st.session_state:
                    expected_cols = st.session_state['expected_columns']
                    missing_cols = [col for col in expected_cols if col not in new_customer.columns]
                    
                    if missing_cols:
                        st.error(f"Missing columns required by the model: {missing_cols}")
                        return
                    
                    # Ensure columns are in the same order as training data
                    new_customer = new_customer[expected_cols]
                
                # Get model and make prediction
                model = st.session_state['model']
                
                try:
                    # Predict
                    churn_pred = model.predict(new_customer)[0]
                    churn_prob = model.predict_proba(new_customer)[0, 1]
                    
                    # Display results
                    st.subheader("Prediction Result")
                    
                    # Display as gauge chart
                    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(polar=True))
                    
                    # Convert probability to angle (0 to 180 degrees)
                    theta = churn_prob * np.pi
                    
                    # Gradient colors for gauge
                    cmap = plt.cm.RdYlGn_r
                    colors = cmap(np.linspace(0, 1, 256))
                    
                    # Draw gauge
                    bars = ax.bar(
                        x=[0], 
                        height=[0.5], 
                        width=[2*np.pi], 
                        bottom=[0.5],
                        color='lightgrey',
                        alpha=0.5
                    )
                    
                    # Draw needle
                    ax.plot([0, theta], [0, 0.9], color='black', linewidth=2)
                    
                    # Add probability text
                    ax.text(0, 0, f"{churn_prob:.1%}", ha='center', va='center', fontsize=24)
                    
                    # Customize gauge appearance
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['polar'].set_visible(False)
                    
                    # Ticks for gauge
                    tick_positions = np.linspace(0, np.pi, 5)
                    tick_labels = [f"{p*100:.0f}%" for p in np.linspace(0, 1, 5)]
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels)
                    
                    # Set title based on prediction
                    title = f"Churn Prediction: {'Yes' if churn_pred == 1 else 'No'}"
                    ax.set_title(title, pad=20, fontsize=16)
                    
                    st.pyplot(fig)
                    
                    # Add interpretation
                    st.subheader("Interpretation")
                    
                    if churn_prob < 0.3:
                        st.success("This customer has a low risk of churning.")
                    elif churn_prob < 0.7:
                        st.warning("This customer has a moderate risk of churning.")
                    else:
                        st.error("This customer has a high risk of churning.")
                    
                    # Provide some insights based on features
                    st.subheader("Potential Insights")
                    
                    insights = []
                    
                    if 'ContractLength' in customer_data and customer_data['ContractLength'] == "Monthly":
                        insights.append("- Monthly contracts have higher churn rates")
                    
                    if 'CustomerServiceCalls' in customer_data and customer_data['CustomerServiceCalls'] > 3:
                        insights.append("- High number of service calls may indicate dissatisfaction")
                    
                    if 'TenureMonths' in customer_data and customer_data['TenureMonths'] < 12:
                        insights.append("- Newer customers are more likely to churn")
                    
                    if insights:
                        for insight in insights:
                            st.write(insight)
                    else:
                        st.write("No specific risk factors identified.")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("This could be due to a mismatch between the model's expected feature set and what was provided.")

# Main app logic
def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    
    # Data Upload page
    if page == "Data Upload":
        st.header("Data Upload")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload your customer data", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                # Load data based on file extension
                if uploaded_file.name.lower().endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Convert churn to binary
                df = convert_churn_to_binary(df)
                
                # Verify columns
                if verify_columns(df):
                    # Save to session state
                    st.session_state['data'] = df
                    
                    # Success message
                    st.success("Data loaded successfully!")
                    
                    # Display data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    
                    # Display data info
                    st.subheader("Data Information")
                    st.write(f"Total records: {df.shape[0]}")
                    st.write(f"Total features: {df.shape[1]}")
                    
                    # Data types
                    st.subheader("Data Types")
                    buffer = pd.DataFrame(df.dtypes, columns=['Data Type'])
                    st.dataframe(buffer)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Data Exploration page
    elif page == "Data Exploration":
        if st.session_state['data'] is not None:
            explore_data(st.session_state['data'])
        else:
            st.warning("Please upload data first.")
    
    # Model Training page
    elif page == "Model Training":
        if st.session_state['data'] is not None:
            build_model(st.session_state['data'])
        else:
            st.warning("Please upload data first.")
    
    # Predictions page
    elif page == "Predictions":
        make_predictions()

if __name__ == "__main__":
    main()