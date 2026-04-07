# Smart AI Plug — Backend

FastAPI backend with a production-trained **Random Forest ML Model** for IoT Smart Plug appliance health prediction.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

pip install fastapi uvicorn scikit-learn pandas numpy joblib
```

## Train the ML Model

```bash
python train_model.py
```
This will generate `appliance_data.csv` (10,000 samples) and save `rf_model.joblib`.

## Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/api/predict` | ML-based appliance fault prediction |

## POST `/api/predict` Payload

```json
{
  "name": "AC Unit",
  "type": "Air Conditioner (AC)",
  "ratedPower": 1500,
  "ratedVoltage": 230,
  "usageHours": 8,
  "voltage": 230,
  "current": 5.1,
  "power": 1100,
  "energy": 2.3,
  "frequency": 50,
  "temperature": 34,
  "pf": 0.92
}
```
# SMART-PLUG
