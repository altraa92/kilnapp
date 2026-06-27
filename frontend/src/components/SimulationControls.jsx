export default function SimulationControls({ isAnalyzing, onAnalyze, onLoadDemo, onReset, t }) {
  return (
    <div className="action-controls">
      <button className="primary-action" type="button" onClick={onAnalyze} disabled={isAnalyzing}>
        {isAnalyzing ? t.analyzingBatch : t.analyzeBatch}
      </button>
      <button type="button" onClick={onLoadDemo}>
        {t.loadTestBatch}
      </button>
      <button type="button" onClick={onReset}>
        {t.reset}
      </button>
    </div>
  );
}
