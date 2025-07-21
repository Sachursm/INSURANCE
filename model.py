import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def train_and_save_model():
    # Load dataset
    insurance_dataset = pd.read_csv('C:/Users/sachu/Desktop/insurance/insurance.csv')
    
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
    
    # Save model
    with open('insurance_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return model

def load_model():
    try:
        with open('insurance_model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        model = train_and_save_model()
    return model