import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_and_save_model():
    try:
        # Load dataset
        dataset_path = 'C:/Users/sachu/Desktop/insurance/insurance.csv'
        print(f"Attempting to load dataset from: {dataset_path}")
        insurance_dataset = pd.read_csv(dataset_path)
        print("Dataset loaded successfully")
        
        # Encoding categorical variables
        insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
        insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
        insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)
        
        # Prepare features and target
        X = insurance_dataset.drop(columns='charges', axis=1)
        Y = insurance_dataset['charges']
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, Y_train)
        print("Model trained successfully")
        
        # Save model
        pkl_path = 'insurance_model.pkl'
        with open(pkl_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to: {os.path.abspath(pkl_path)}")
        
        return model
    except Exception as e:
        print(f"Error in train_and_save_model: {str(e)}")
        raise

def load_model():
    try:
        pkl_path = 'insurance_model.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as file:
                model = pickle.load(file)
            print("Model loaded successfully")
        else:
            print("No existing model found, training new model")
            model = train_and_save_model()
        return model
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        raise

if __name__ == "__main__":
    print("Running model.py directly")
    model = load_model()
    print("Script completed")