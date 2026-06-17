# Intelligent Solar-Assisted Fish Kiln Prototype

Working prototype for a final year project dashboard and API:

**Development of an Intelligent Solar-Assisted Fish Kiln System with Sensor-Based Monitoring, Predictive Drying-Time Estimation, and Multilingual Web Application.**

The project contains:

- `backend/`: Flask API, synthetic data generation, Random Forest Regression training, and prediction endpoints.
- `frontend/`: Vite + React multilingual monitoring dashboard with simulated sensor updates.

The prototype does not connect to ESP32 hardware yet. It uses simulated inputs and a synthetic dataset to demonstrate the intended workflow. Real experimental drying data can replace `backend/synthetic_drying_data.csv` later.

## Quick Start

Backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Set the deployed frontend environment variable:

```bash
VITE_API_BASE_URL=https://your-backend-url
```

Backend deployment target: Render or Railway.
Frontend deployment target: Vercel or Netlify.
