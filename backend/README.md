# Fish Kiln Backend

Flask API and Random Forest Regression model for the intelligent solar-assisted fish kiln prototype.

The current model is trained on synthetic drying data so the full system workflow can be demonstrated before ESP32 sensor data is available. In the final implementation, `synthetic_drying_data.csv` should be replaced with real drying data collected from the sensor network.

## Local Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the API:

```bash
python app.py
```

The backend starts on `http://127.0.0.1:5000`.

## Endpoints

- `GET /api/health` checks service status.
- `POST /api/predict` computes drying metrics and predicts remaining drying time.
- `GET /api/model-info` returns model type, feature names, metrics, and the synthetic-data notice.
- `GET /api/sample-data` returns sample rows from the generated synthetic dataset.

## Example Prediction Request

```json
{
  "fishType": "catfish",
  "initialWeight": 2.0,
  "currentWeight": 1.4,
  "temperature": 60,
  "humidity": 45,
  "elapsedTime": 120
}
```

## Deployment Notes

The backend can be deployed on Render or Railway.

For this prototype, Render is the simplest free PaaS option. This repository includes a root-level `render.yaml` Blueprint that deploys this backend from the `backend/` directory.

Render settings if configuring manually:

- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt && python train_model.py`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
- Health Check Path: `/api/health`
- Environment Variable: `PYTHON_VERSION=3.12.13`

Suggested start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

If `model.pkl` is not present when the Flask app starts, the app automatically runs `train_model.py` to generate the model and synthetic dataset.
