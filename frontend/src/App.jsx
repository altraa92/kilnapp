import { useCallback, useEffect, useMemo, useState } from "react";
import AlertsPanel from "./components/AlertsPanel.jsx";
import InputPanel from "./components/InputPanel.jsx";
import MetricCard from "./components/MetricCard.jsx";
import ModelInfo from "./components/ModelInfo.jsx";
import SimulationControls from "./components/SimulationControls.jsx";
import { API_BASE_URL, getModelInfo, predictDrying } from "./services/api.js";
import { languages, translate, translations } from "./translations";

const DEFAULT_FORM = {
  fishType: "catfish",
  initialWeight: 2.0,
  currentWeight: 2.0,
  temperature: 60,
  humidity: 45,
  elapsedTime: 0,
};

const INITIAL_MOISTURE = {
  catfish: 65,
  tilapia: 80,
  mackerel: 75,
};

const TARGET_MOISTURE = 15;

function formatNumber(value, places = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: places,
  });
}

function targetWeightFor(fishType, initialWeight) {
  const initialMoisture = INITIAL_MOISTURE[fishType] || INITIAL_MOISTURE.catfish;
  const dryMatter = Number(initialWeight) * (1 - initialMoisture / 100);
  return dryMatter / (1 - TARGET_MOISTURE / 100);
}

function statusTone(status) {
  if (status === "Drying complete") return "complete";
  if (status === "Near target moisture") return "warning";
  if (status === "Invalid weight reading") return "danger";
  return "monitoring";
}

export default function App() {
  const [language, setLanguage] = useState("en");
  const [form, setForm] = useState(DEFAULT_FORM);
  const [prediction, setPrediction] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");

  const t = translations[language];

  const requestPrediction = useCallback(async (values) => {
    try {
      const data = await predictDrying(values);
      setPrediction(data);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    }
  }, []);

  useEffect(() => {
    getModelInfo()
      .then((data) => setModelInfo(data))
      .catch((requestError) => setError(requestError.message));
  }, []);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      requestPrediction(form);
    }, 250);

    return () => window.clearTimeout(timeout);
  }, [form, requestPrediction]);

  useEffect(() => {
    if (prediction?.dryingStatus === "Drying complete") {
      setIsRunning(false);
    }
  }, [prediction]);

  useEffect(() => {
    if (!isRunning) {
      return undefined;
    }

    const interval = window.setInterval(() => {
      setForm((current) => {
        const targetWeight = targetWeightFor(current.fishType, current.initialWeight);
        const currentWeight = Number(current.currentWeight);
        const temperature = Number(current.temperature);
        const humidity = Number(current.humidity);
        const heatLift = Math.max(0, Math.min(1, (temperature - 45) / 25));
        const humidityDrag = Math.max(0, Math.min(1, (humidity - 25) / 60));
        const rate = 0.05 + heatLift * 0.035 - humidityDrag * 0.018;
        const stepLoss = Math.max((currentWeight - targetWeight) * rate, 0.008);
        const nextWeight = Math.max(targetWeight, currentWeight - stepLoss);

        return {
          ...current,
          currentWeight: Number(nextWeight.toFixed(3)),
          elapsedTime: Number(current.elapsedTime) + 15,
        };
      });
    }, 1400);

    return () => window.clearInterval(interval);
  }, [isRunning]);

  const metrics = useMemo(() => {
    const status = prediction?.dryingStatus || "Drying in progress";
    return [
      {
        label: t.fishType,
        value: t[form.fishType],
      },
      {
        label: t.initialMoistureContent,
        value: formatNumber(prediction?.initialMoistureContent ?? INITIAL_MOISTURE[form.fishType]),
        unit: t.percent,
      },
      {
        label: t.currentWeight,
        value: formatNumber(form.currentWeight, 3),
        unit: t.kg,
      },
      {
        label: t.temperature,
        value: formatNumber(form.temperature, 1),
        unit: t.celsius,
        tone: Number(form.temperature) > 70 ? "danger" : "monitoring",
      },
      {
        label: t.humidity,
        value: formatNumber(form.humidity, 1),
        unit: t.percent,
        tone: Number(form.humidity) > 80 ? "warning" : "neutral",
      },
      {
        label: t.estimatedMoistureContent,
        value: formatNumber(prediction?.estimatedMoistureContent),
        unit: t.percent,
        tone: statusTone(status),
      },
      {
        label: t.targetMoistureContent,
        value: formatNumber(prediction?.targetMoistureContent ?? TARGET_MOISTURE),
        unit: t.percent,
      },
      {
        label: t.dryMatter,
        value: formatNumber(prediction?.dryMatter, 3),
        unit: t.kg,
      },
      {
        label: t.weightLoss,
        value: formatNumber(prediction?.weightLoss, 3),
        unit: t.kg,
      },
      {
        label: t.dryingProgress,
        value: formatNumber(prediction?.dryingProgress),
        unit: t.percent,
        tone: statusTone(status),
      },
      {
        label: t.estimatedRemainingTime,
        value: formatNumber(prediction?.estimatedRemainingTime),
        unit: t.minutes,
        tone: "monitoring",
      },
      {
        label: t.dryingStatus,
        value: translate(t, status),
        tone: statusTone(status),
      },
    ];
  }, [form, prediction, t]);

  const progress = Math.max(0, Math.min(100, Number(prediction?.dryingProgress || 0)));
  const translatedStatus = translate(t, prediction?.dryingStatus || "Drying in progress");

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="topbar-title">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <p>{t.dashboard}</p>
            <h1>{t.headerTitle}</h1>
          </div>
        </div>

        <label className="language-select">
          <span>{t.language}</span>
          <span className="select-wrap">
            <select value={language} onChange={(event) => setLanguage(event.target.value)}>
              {languages.map((item) => (
                <option key={item.code} value={item.code}>
                  {item.label}
                </option>
              ))}
            </select>
          </span>
        </label>
      </header>

      <section className="control-row">
        <InputPanel form={form} onChange={setForm} t={t} />
        <div className="status-panel">
          <div className="status-cluster">
            <div className="progress-dial" style={{ "--progress": `${progress * 3.6}deg` }}>
              <div className="dial-readout">
                <strong>{formatNumber(progress)}</strong>
                <span>{t.percent}</span>
              </div>
            </div>

            <div className="status-copy">
              <div className={`status-badge ${statusTone(prediction?.dryingStatus)}`}>
                {translatedStatus}
              </div>
              <p>
                {t.estimatedMoistureContent}: {formatNumber(prediction?.estimatedMoistureContent)}
                {t.percent}
              </p>
            </div>
          </div>

          <div className="progress-block">
            <div className="progress-copy">
              <span>{t.dryingProgress}</span>
              <strong>{formatNumber(progress)}{t.percent}</strong>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
          <SimulationControls
            isRunning={isRunning}
            onStart={() => setIsRunning(true)}
            onPause={() => setIsRunning(false)}
            onReset={() => {
              setIsRunning(false);
              setForm(DEFAULT_FORM);
            }}
            t={t}
          />
          {error ? <p className="error-text">{error}</p> : null}
        </div>
      </section>

      <section className="metrics-grid">
        {metrics.map((metric) => (
          <MetricCard
            key={metric.label}
            label={metric.label}
            value={metric.value}
            unit={metric.unit}
            tone={metric.tone}
          />
        ))}
      </section>

      <section className="lower-grid">
        <AlertsPanel alerts={prediction?.alerts} t={t} />
        <ModelInfo modelInfo={modelInfo} t={t} apiBaseUrl={API_BASE_URL} />
      </section>
    </main>
  );
}
