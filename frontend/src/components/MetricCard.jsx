export default function MetricCard({ label, value, unit, tone = "neutral" }) {
  return (
    <article className={`metric-card ${tone}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">
        {value}
        {unit ? <small>{unit}</small> : null}
      </strong>
    </article>
  );
}
