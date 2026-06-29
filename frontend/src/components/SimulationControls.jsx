export default function SimulationControls({
  isAnalyzing,
  isLiveFeed,
  onAnalyze,
  onLoadDemo,
  onStartFeed,
  onStopFeed,
  onReset,
  t,
}) {
  return (
    <div className="action-controls">
      <button className="primary-action" type="button" onClick={onAnalyze} disabled={isAnalyzing}>
        {isAnalyzing ? t.analyzingBatch : t.analyzeBatch}
      </button>
      <button type="button" onClick={isLiveFeed ? onStopFeed : onStartFeed} disabled={isAnalyzing}>
        {isLiveFeed ? t.stopDemoFeed : t.startDemoFeed}
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
