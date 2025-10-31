# model_improved_randomforest.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor  # <-- Switched to Random Forest for better accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score  # <-- Added for R² evaluation
import pickle
import os
from scipy.stats import uniform, randint, loguniform

def train_and_save_model():
    try:
        # -------------------------------------------------
        # 1. Load dataset
        # -------------------------------------------------
        dataset_path = 'C:/Users/sachu/Desktop/insurance/INSURANCE/insurance.csv'  # Original local path
        #dataset_path = 'https://raw.githubusercontent.com/datagy/data/main/insurance.csv'  # Public URL for portability
        print(f"Attempting to load dataset from: {dataset_path}")
        insurance_dataset = pd.read_csv(dataset_path)
        print("Dataset loaded successfully")
        print(f"Dataset shape: {insurance_dataset.shape}")

        # -------------------------------------------------
        # 2. Encode categorical variables
        # -------------------------------------------------
        insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
        insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
        insurance_dataset.replace(
            {'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}},
            inplace=True,
        )

        # -------------------------------------------------
        # 3. Prepare features / target
        # -------------------------------------------------
        X = insurance_dataset.drop(columns='charges', axis=1)
        y = insurance_dataset['charges']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2
        )

        # -------------------------------------------------
        # 4. Build a pipeline (scaling + Random Forest)
        # -------------------------------------------------
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Optional for trees but keeps consistency
                ("rf", RandomForestRegressor(random_state=42)),
            ]
        )

        # -------------------------------------------------
        # 5. Hyper-parameter search space (tailored for Random Forest)
        # -------------------------------------------------
        param_distributions = {
            "rf__n_estimators": randint(50, 200),  # Number of trees
            "rf__max_depth": [3, 5, 7, 10, None],  # Max depth (None for unlimited)
            "rf__min_samples_split": randint(2, 11),  # Min samples to split
            "rf__min_samples_leaf": randint(1, 5),  # Min samples at leaf
            "rf__max_features": ["sqrt", "log2", None],  # Features per split
        }

        # -------------------------------------------------
        # 6. RandomizedSearchCV
        # -------------------------------------------------
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=150,  # Increased for better tuning
            cv=5,  # 5-fold CV
            scoring="r2",  # Optimize for R² directly
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            verbose=1,
        )

        print("Starting hyper-parameter tuning with RandomizedSearchCV...")
        random_search.fit(X_train, y_train)
        print("Tuning finished!")

        # -------------------------------------------------
        # 7. Best model & results
        # -------------------------------------------------
        best_model = random_search.best_estimator_
        print("\nBest parameters found:")
        for k, v in random_search.best_params_.items():
            print(f"  {k}: {v}")
        print(f"Best CV R²: {random_search.best_score_:.4f}")

        # -------------------------------------------------
        # 8. Evaluate on hold-out test set
        # -------------------------------------------------
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(((test_pred - y_test) ** 2).mean())
        
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.2f}")

        # -------------------------------------------------
        # 9. Save the *best* model
        # -------------------------------------------------
        pkl_path = "insurance_model_improved.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"Best model saved to: {os.path.abspath(pkl_path)}")

        return best_model

    except Exception as e:
        print(f"Error in train_and_save_model: {str(e)}")
        raise


def load_model():
    """Load a saved model; if none exists, train a new one with tuning."""
    try:
        pkl_path = "insurance_model_improved.pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            print("Model loaded successfully")
        else:
            print("No existing model found, training a new one with hyper-parameter tuning...")
            model = train_and_save_model()
        return model
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        raise


if __name__ == "__main__":
    print("Running model_improved_randomforest.py directly")
    model = load_model()
    print("Script completed")