const fishOptions = ["catfish", "tilapia", "mackerel"];

export default function InputPanel({ form, onChange, t }) {
  const updateField = (event) => {
    const { name, value } = event.target;
    onChange({
      ...form,
      [name]: name === "fishType" ? value : Number(value),
    });
  };

  return (
    <section className="panel input-panel">
      <div className="section-heading">
        <span>{t.dashboard}</span>
        <h2>{t.fishBatchInput}</h2>
      </div>

      <div className="form-grid">
        <label className="field">
          <span>{t.fishType}</span>
          <span className="select-wrap">
            <select name="fishType" value={form.fishType} onChange={updateField}>
              {fishOptions.map((type) => (
                <option key={type} value={type}>
                  {t[type]}
                </option>
              ))}
            </select>
          </span>
        </label>

        <label className="field">
          <span>{t.initialWeight} ({t.kg})</span>
          <input
            name="initialWeight"
            type="number"
            min="0.1"
            step="0.1"
            value={form.initialWeight}
            onChange={updateField}
          />
        </label>

        <label className="field">
          <span>{t.currentWeight} ({t.kg})</span>
          <input
            name="currentWeight"
            type="number"
            min="0.1"
            step="0.01"
            value={form.currentWeight}
            onChange={updateField}
          />
        </label>

        <label className="field">
          <span>{t.temperature} ({t.celsius})</span>
          <input
            name="temperature"
            type="number"
            min="0"
            step="1"
            value={form.temperature}
            onChange={updateField}
          />
        </label>

        <label className="field">
          <span>{t.humidity} ({t.percent})</span>
          <input
            name="humidity"
            type="number"
            min="0"
            max="100"
            step="1"
            value={form.humidity}
            onChange={updateField}
          />
        </label>

        <label className="field">
          <span>{t.elapsedTime} ({t.minutes})</span>
          <input
            name="elapsedTime"
            type="number"
            min="0"
            step="5"
            value={form.elapsedTime}
            onChange={updateField}
          />
        </label>
      </div>
    </section>
  );
}
