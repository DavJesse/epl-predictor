import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import os

# Paths
DATA_PATH = os.path.join("data", "processed", "epl_training_data.csv")
MODEL_PATH = os.path.join("models_registry", "outcome_model_v1.pkl")

def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Drop rows with missing values (the first 5 games of every season usually have NaNs for form)
    df = df.dropna()
    
    # 2. Define our "Features" (X) and "Target" (y)
    # We want to predict 'Result' (H, D, A) using the Form metrics
    features = [
        'Home_Points_Avg', 'Away_Points_Avg', 
        'Home_Goals_Avg', 'Away_Goals_Avg',
        'Home_Conceded_Avg', 'Away_Conceded_Avg'
    ]
    target = 'FTR' # H (Home Win), D (Draw), A (Away Win)
    
    X = df[features]
    y = df[target]
    
    # 3. Split Data (Standard Split)
    # Ideally we split by DATE (train on past, test on future), but for a first test, 
    # random split is okay to just see if the code works.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   Training on {len(X_train)} matches, testing on {len(X_test)} matches.")
    
    # 4. Initialize and Train the Model
    # n_estimators=100 means "create 100 decision trees"
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nDetailed Report:")
    print(classification_report(y_test, predictions))
    
    # 6. Save the model so we can use it later
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()