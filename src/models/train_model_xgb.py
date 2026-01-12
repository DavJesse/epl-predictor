import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Paths
DATA_PATH = os.path.join("data", "processed", "epl_training_data.csv")
MODEL_PATH = os.path.join("models_registry", "outcome_model_xgb.json") # XGBoost prefers .json
ENCODER_PATH = os.path.join("models_registry", "label_encoder.pkl")

def train():
    print("Loading data for XGBoost...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Data not found.")
        return

    # 1. Clean Data
    df = df.dropna()
    
    # 2. Define Features
    features = [
        'Home_Points_Avg', 'Away_Points_Avg', 
        'Home_Goals_Avg', 'Away_Goals_Avg',
        'Home_Conceded_Avg', 'Away_Conceded_Avg'
    ]
    target = 'FTR'
    
    X = df[features]
    y = df[target]
    
    # 3. Label Encoding (Critical for XGBoost)
    # Converts 'A' -> 0, 'D' -> 1, 'H' -> 2
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save the encoder so we can decode predictions later (0 -> 'A')
    joblib.dump(le, ENCODER_PATH)
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # 5. Initialize XGBoost
    # learning_rate: How much each tree contributes (lower = slower but more precise)
    # max_depth: How complex each tree can be
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05, 
        max_depth=3, 
        random_state=42
    )
    
    print("   Training XGBoost Model...")
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nXGBoost Accuracy: {accuracy:.2%}")
    
    # We need to map the numbers (0,1,2) back to names (Away, Draw, Home) for the report
    target_names = le.classes_ # e.g. ['A', 'D', 'H']
    print("\nDetailed Report:")
    print(classification_report(y_test, predictions, target_names=target_names))
    
    # 7. Save
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()