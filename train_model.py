import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def generate_dataset(num_samples=10000):
    np.random.seed(42)
    data = []
    
    # 60% Normal (Optimal performance)
    for _ in range(int(num_samples * 0.6)):
        v = np.random.normal(230, 5) # Close to 230V
        i = np.random.normal(4,  1) # Normal current
        rp = 1000
        p = np.random.normal(v * i * 0.95, 20) # High PF ~ 0.95
        t = np.random.normal(32, 4) # Normal temperature
        data.append({"voltage": v, "current": i, "power": p, "temperature": t, "ratedPower": rp, "fault_status": 0})

    # 25% Warning (Sub-optimal, age deterioration)
    for _ in range(int(num_samples * 0.25)):
        v = np.random.normal(245, 8) if np.random.choice([True, False]) else np.random.normal(215, 8)
        i = np.random.normal(6, 1.5)
        rp = 1000
        p = np.random.normal(v * i * 0.85, 40) # Lower PF ~ 0.85
        t = np.random.normal(45, 5) # Elevated temp
        data.append({"voltage": v, "current": i, "power": p, "temperature": t, "ratedPower": rp, "fault_status": 1})

    # 15% Faulty (Critical, very poor performance or dangerous levels)
    for _ in range(int(num_samples * 0.15)):
        v = np.random.normal(190, 15) if np.random.choice([True, False]) else np.random.normal(260, 15)
        i = np.random.normal(8, 2.5) # Overcurrent
        rp = 1000
        p = np.random.normal(v * i * 0.65, 80) # Poor PF ~ 0.65
        t = np.random.normal(65, 8) # Dangerous temp
        data.append({"voltage": v, "current": i, "power": p, "temperature": t, "ratedPower": rp, "fault_status": 2})

    df = pd.DataFrame(data)
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "appliance_data.csv")
    model_path = os.path.join(base_dir, "rf_model.joblib")

    print(f"Generating realistic synthetic dataset with 10,000 samples...")
    df = generate_dataset(10000)
    df.to_csv(data_path, index=False)
    print(f"Dataset saved to {data_path}")
    
    print("Training Production Random Forest Model...")
    X = df[["voltage", "current", "power", "temperature", "ratedPower"]]
    y = df["fault_status"]
    
    # Complex production-ready model
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    
    # Save the model
    joblib.dump(rf_model, model_path)
    print(f"Model successfully saved to {model_path}!")
