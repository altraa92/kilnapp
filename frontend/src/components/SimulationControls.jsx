export default function SimulationControls({ isRunning, onStart, onPause, onReset, t }) {
  return (
    <div className="simulation-controls">
      <button className="primary-action" type="button" onClick={onStart} disabled={isRunning}>
        {t.startSimulation}
      </button>
      <button type="button" onClick={onPause} disabled={!isRunning}>
        {t.pauseSimulation}
      </button>
      <button type="button" onClick={onReset}>
        {t.reset}
      </button>
    </div>
  );
}
