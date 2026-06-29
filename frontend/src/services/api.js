const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }

  return data;
}

export function predictDrying(payload) {
  return request("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function startDemoFeed(payload) {
  return request("/api/demo/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function stopDemoFeed() {
  return request("/api/demo/stop", {
    method: "POST",
  });
}

export function getLatestReading() {
  return request("/api/readings/latest");
}

export function submitHardwareReading(payload) {
  return request("/api/readings", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getModelInfo() {
  return request("/api/model-info");
}

export { API_BASE_URL };
