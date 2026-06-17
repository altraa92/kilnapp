# Fish Kiln Frontend

Vite + React dashboard for the intelligent fish kiln prototype.

The frontend demonstrates the final workflow with simulated sensor readings. It sends fish type, weight, temperature, humidity, and elapsed time to the Flask backend, then displays estimated moisture content, drying progress, alerts, and Random Forest remaining-time prediction.

## Local Setup

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal.

For local development, the frontend defaults to:

```bash
http://127.0.0.1:5000
```

To point the frontend to a deployed backend, create `.env`:

```bash
VITE_API_BASE_URL=https://your-backend-url
```

For a Render backend, the URL usually looks like:

```bash
VITE_API_BASE_URL=https://fish-kiln-backend.onrender.com
```

When deploying this Vite app to Vercel, set the same variable in the Vercel project settings:

- Name: `VITE_API_BASE_URL`
- Value: your Render backend URL, for example `https://fish-kiln-backend.onrender.com`
- Environments: Production, Preview, and Development if you want all Vercel builds to use the deployed backend

## Deployment Notes

The frontend can be deployed on Vercel or Netlify.

Set this environment variable in the deployment platform:

```bash
VITE_API_BASE_URL=https://your-backend-url
```

## Important Prototype Note

The backend model is currently trained on synthetic data for demonstration. In the final project implementation, the synthetic dataset will be replaced with real experimental drying data collected from the ESP32 sensor network.
