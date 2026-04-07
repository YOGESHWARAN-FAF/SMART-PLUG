from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import datetime
import joblib
import os

app = FastAPI(title="Smart AI Plug Backend", version="1.0")

# Setup CORS to allow React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Loading the Production Environment Model ---
print("Initializing AI Application...")
model_path = os.path.join(os.path.dirname(__file__), "rf_model.joblib")

if os.path.exists(model_path):
    print(f"Loading Production Random Forest Model from {model_path}...")
    rf_model = joblib.load(model_path)
    print("Model loaded successfully. AI service active.")
else:
    print(f"CRITICAL WARNING: Production ML Model not found at {model_path}!")
    print("Please run train_model.py first to generate real synthetic dataset and compile joblib schema.")
    print("Falling back to dummy inline model initialization (Not Recommended)...")
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Give it dummy data so the predict endpoint doesn't crash entirely.
    X_dummy = np.array([[230, 5, 1100, 35, 1200], [250, 6, 1400, 45, 1200], [200, 8, 1800, 65, 1200]])
    y_dummy = np.array([0, 1, 2])
    rf_model.fit(X_dummy, y_dummy)
    print("Fallback Model initialized.")

# --- API Models ---
class PredictionRequest(BaseModel):
    name: str
    type: str
    ratedPower: float
    ratedVoltage: float
    usageHours: float
    voltage: float
    current: float
    power: float
    energy: float
    frequency: float
    temperature: float
    pf: float

@app.post("/api/predict")
async def predict_health(data: PredictionRequest):
    # Calculate Efficiency
    efficiency = 0
    if data.power > 0 and data.ratedPower > 0:
        efficiency = (data.power / data.ratedPower) * 100
        if efficiency > 100: 
            efficiency = max(0, 95 - (efficiency - 100)) # Penalty
    efficiency = min(max(efficiency, 0), 100)

    # Use Random Forest to predict Fault Status
    features = np.array([[data.voltage, data.current, data.power, data.temperature, data.ratedPower]])
    fault_prediction = rf_model.predict(features)[0]
    
    status_map = {0: "Normal", 1: "Warning", 2: "Faulty"}
    fault_status = status_map[fault_prediction]
    
    # Detailed Health Score Generation
    temp_penalty = max(0, (data.temperature - 40) * 1.5)
    pf_penalty = max(0, (0.8 - data.pf) * 100) if data.pf > 0 else 0
    voltage_diff = abs(data.voltage - data.ratedVoltage)
    voltage_penalty = max(0, voltage_diff - 15)
    
    health = 100 - temp_penalty - pf_penalty - voltage_penalty
    health = min(max(health, 0), 100)

    # Contextual Explanations
    if fault_status == "Faulty":
         maintenance_msg = "Critical failure predicted or occurring. Immediate maintenance required on coils or bearings."
    elif fault_status == "Warning":
         maintenance_msg = "Sub-optimal operation detected. Schedule maintenance soon to prevent hardware degradation."
    else:
         maintenance_msg = "Appliance is functioning optimally based on machine learning classifications. Routine checks every 6 months."
    
    energy_analysis = "Power consumption within expected statistical boundaries."
    if efficiency < 70:
        energy_analysis = "High energy wastage detected (efficiency <70%). Check for electrical leakage or age deterioration."
    elif efficiency > 90:
        energy_analysis = "Excellent energy utilization. Very minimal power loss reported."

    temp_warning = "Operating temperature is nominal."
    if data.temperature > 50:
        temp_warning = "Critical Overheating Warning. Risk of core meltdown or fire hazards."
    elif data.temperature > 40:
        temp_warning = "Elevated Temperature Warning. Monitor cooling mechanisms."

    return {
        "efficiency": round(efficiency, 1),
        "health": round(health, 1),
        "faultStatus": fault_status,
        "energyAnalysis": energy_analysis,
        "tempWarning": temp_warning,
        "maintenanceMsg": maintenance_msg,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/")
def root():
    return {"status": "Backend Active", "model": "Random Forest Classifier - Optimized for IoT Plugs"}
