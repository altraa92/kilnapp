import { translate } from "../translations";

export default function AlertsPanel({ alerts, t }) {
  const activeAlerts = alerts || [];

  return (
    <section className="panel alerts-panel">
      <div className="section-heading">
        <span>{t.dashboard}</span>
        <h2>{t.alerts}</h2>
      </div>

      {activeAlerts.length === 0 ? (
        <p className="empty-alert">{t.noAlerts}</p>
      ) : (
        <div className="alert-list">
          {activeAlerts.map((alert) => (
            <div className="alert-item" key={alert}>
              {translate(t, alert)}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
