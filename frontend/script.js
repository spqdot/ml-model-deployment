// Replace with your deployed API URL (e.g. https://your-api.onrender.com)
// For local development use: http://127.0.0.1:8000
const API_BASE = "http://127.0.0.1:8000";

const form       = document.getElementById("prediction-form");
const submitBtn  = document.getElementById("submit-btn");
const resultDiv  = document.getElementById("result");
const resultVal  = document.getElementById("result-value");
const resultMeta = document.getElementById("result-meta");
const errorDiv   = document.getElementById("error");
const errorMsg   = document.getElementById("error-message");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessages();
  setLoading(true);

  const data = new FormData(form);

  const record = {
    store_ID:            parseInt(data.get("store_ID"), 10),
    day_of_week:         parseInt(data.get("day_of_week"), 10),
    nb_customers_on_day: parseInt(data.get("nb_customers_on_day"), 10),
    open:                parseInt(data.get("open"), 10),
    promotion:           parseInt(data.get("promotion"), 10),
    state_holiday:       data.get("state_holiday"),
    school_holiday:      parseInt(data.get("school_holiday"), 10),
  };

  const payload = {
    records: [record],
    model_name: data.get("model_name"),
  };

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const json = await response.json();

    if (!response.ok) {
      showError(json.detail || `Server error: ${response.status}`);
      return;
    }

    const prediction = json.predictions[0];
    resultVal.textContent  = `€ ${prediction.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    resultMeta.textContent = `Model: ${json.model_used}  ·  Records: ${json.num_records}`;
    resultDiv.classList.remove("hidden");

  } catch (err) {
    showError("Could not reach the API. Make sure the server is running (uvicorn app:app --reload).");
  } finally {
    setLoading(false);
  }
});

function clearMessages() {
  resultDiv.classList.add("hidden");
  errorDiv.classList.add("hidden");
}

function showError(message) {
  errorMsg.textContent = message;
  errorDiv.classList.remove("hidden");
}

function setLoading(loading) {
  submitBtn.disabled    = loading;
  submitBtn.textContent = loading ? "Predicting…" : "Predict Sales";
}
