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

const TEST_BATCH = {
  fishType: "catfish",
  initialWeight: 2.0,
  currentWeight: 1.4,
  temperature: 60,
  humidity: 45,
  elapsedTime: 120,
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
  const dryMatter = Number(initialWeight || 0) * (1 - initialMoisture / 100);
  return dryMatter / (1 - TARGET_MOISTURE / 100);
}

function statusTone(status) {
  if (status === "Drying complete") return "complete";
  if (status === "Near target moisture") return "warning";
  if (status === "Invalid weight reading") return "danger";
  if (status === "Awaiting analysis") return "idle";
  return "monitoring";
}

function buildMoisturePath(initialMoisture, currentMoisture, targetMoisture) {
  const values = [
    Number(initialMoisture || 0),
    Number(currentMoisture ?? initialMoisture ?? 0),
    Number(targetMoisture || TARGET_MOISTURE),
  ];
  const max = Math.max(...values, 100);

  return values.map((value, index) => ({
    x: 14 + index * 86,
    y: 94 - (value / max) * 72,
  }));
}

export default function App() {
  const [language, setLanguage] = useState("en");
  const [form, setForm] = useState(DEFAULT_FORM);
  const [prediction, setPrediction] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");

  const t = translations[language];

  const requestPrediction = useCallback(async (values) => {
    try {
      setIsAnalyzing(true);
      const data = await predictDrying(values);
      setPrediction(data);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  useEffect(() => {
    getModelInfo()
      .then((data) => setModelInfo(data))
      .catch((requestError) => setError(requestError.message));
  }, []);

  const metrics = useMemo(() => {
    const status = prediction?.dryingStatus || "Awaiting analysis";
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
  const status = prediction?.dryingStatus || "Awaiting analysis";
  const translatedStatus = translate(t, status);
  const moisturePoints = buildMoisturePath(
    prediction?.initialMoistureContent ?? INITIAL_MOISTURE[form.fishType],
    prediction?.estimatedMoistureContent,
    prediction?.targetMoistureContent ?? TARGET_MOISTURE
  );
  const moisturePath = moisturePoints.map((point) => `${point.x},${point.y}`).join(" ");
  const modelMetrics = modelInfo?.metrics || {};
  const targetWeight = targetWeightFor(form.fishType, form.initialWeight);

  return (
    <main className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <strong>KilnAI</strong>
            <span>{t.monitoringConsole}</span>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Dashboard sections">
          <a href="#batch">{t.batch}</a>
          <a href="#prediction">{t.prediction}</a>
          <a href="#model">{t.modelInformation}</a>
          <a href="#alerts">{t.alerts}</a>
        </nav>

        <div className="sidebar-footer">
          <span>{t.backend}</span>
          <strong>{API_BASE_URL.replace(/^https?:\/\//, "")}</strong>
        </div>
      </aside>

      <section className="app-shell">
        <header className="topbar">
          <div className="topbar-title">
            <p>{t.dashboard}</p>
            <h1>{t.headerTitle}</h1>
            <span>{t.operatorSubtitle}</span>
          </div>

          <div className="topbar-actions">
            <div className="connection-pill">
              <span className="live-dot" aria-hidden="true" />
              {t.apiOnline}
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
          </div>
        </header>

        <section className="overview-grid" aria-label={t.metrics}>
          <article className="overview-card">
            <span>{t.estimatedRemainingTime}</span>
            <strong>{formatNumber(prediction?.estimatedRemainingTime)}</strong>
            <small>{t.minutes}</small>
          </article>
          <article className="overview-card">
            <span>{t.estimatedMoistureContent}</span>
            <strong>{formatNumber(prediction?.estimatedMoistureContent)}</strong>
            <small>{t.percent}</small>
          </article>
          <article className="overview-card">
            <span>{t.dryingProgress}</span>
            <strong>{formatNumber(progress)}</strong>
            <small>{t.percent}</small>
          </article>
          <article className="overview-card">
            <span>{t.modelAccuracy}</span>
            <strong>{formatNumber(modelMetrics.r2, 3)}</strong>
            <small>R2</small>
          </article>
        </section>

        <section className="control-row" id="batch">
          <InputPanel form={form} onChange={setForm} t={t} />

          <div className="status-panel" id="prediction">
            <div className="status-cluster">
              <div className="progress-dial" style={{ "--progress": `${progress * 3.6}deg` }}>
                <div className="dial-readout">
                  <strong>{formatNumber(progress)}</strong>
                  <span>{t.percent}</span>
                </div>
              </div>

              <div className="status-copy">
                <div className={`status-badge ${statusTone(status)}`}>{translatedStatus}</div>
                <p>
                  {t.estimatedMoistureContent}: {formatNumber(prediction?.estimatedMoistureContent)}
                  {t.percent}
                </p>
              </div>
            </div>

            <div className="forecast-chart">
              <div className="chart-head">
                <span>{t.moistureTrajectory}</span>
                <strong>
                  {formatNumber(targetWeight, 3)} {t.kg} {t.targetWeight}
                </strong>
              </div>
              <svg viewBox="0 0 200 112" role="img" aria-label="Moisture trajectory">
                <line className="axis" x1="14" y1="94" x2="186" y2="94" />
                <line className="axis" x1="14" y1="20" x2="14" y2="94" />
                <polyline className="path" points={moisturePath} />
                {moisturePoints.map((point, index) => (
                  <circle key={`${point.x}-${index}`} cx={point.x} cy={point.y} r="3.4" />
                ))}
              </svg>
            </div>

            <div className="progress-block">
              <div className="progress-copy">
                <span>{t.dryingProgress}</span>
                <strong>
                  {formatNumber(progress)}
                  {t.percent}
                </strong>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
            </div>

            <SimulationControls
              isAnalyzing={isAnalyzing}
              onAnalyze={() => requestPrediction(form)}
              onLoadDemo={() => {
                setForm(TEST_BATCH);
                requestPrediction(TEST_BATCH);
              }}
              onReset={() => {
                setForm(DEFAULT_FORM);
                setPrediction(null);
                setError("");
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
          <div id="alerts">
            <AlertsPanel alerts={prediction?.alerts} t={t} />
          </div>
          <div id="model">
            <ModelInfo modelInfo={modelInfo} t={t} apiBaseUrl={API_BASE_URL} />
          </div>
        </section>
      </section>
    </main>
  );
}
