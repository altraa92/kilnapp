export default function ModelInfo({ modelInfo, t, apiBaseUrl }) {
  const metrics = modelInfo?.metrics;
  const features = modelInfo?.features || [];

  return (
    <section className="panel model-panel">
      <div className="section-heading">
        <span>{t.syntheticDataModel}</span>
        <h2>{t.modelInformation}</h2>
      </div>

      <dl className="model-list">
        <div>
          <dt>{t.modelType}</dt>
          <dd>{modelInfo?.modelType || "Random Forest Regression"}</dd>
        </div>
        <div>
          <dt>{t.trainingData}</dt>
          <dd>{modelInfo?.trainingDataType || "synthetic"}</dd>
        </div>
        <div>
          <dt>{t.backend}</dt>
          <dd>{apiBaseUrl}</dd>
        </div>
      </dl>

      {metrics ? (
        <div className="metric-strip">
          <span>{t.mae}: {metrics.mae}</span>
          <span>{t.rmse}: {metrics.rmse}</span>
          <span>{t.r2}: {metrics.r2}</span>
        </div>
      ) : null}

      <p className="model-note">{modelInfo?.message || t.syntheticNotice}</p>

      <details className="feature-details">
        <summary>{t.features}</summary>
        <p>{features.join(", ")}</p>
      </details>
    </section>
  );
}
