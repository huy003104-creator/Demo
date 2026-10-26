const $ = (id) => document.getElementById(id);
const feed = $("cameraFeed");
let pollTimer = null;

function setRunning(running) {
  $("startCamera").disabled = running;
  $("startVideo").disabled = running;
  $("stop").disabled = !running;
  $("feedBadge").textContent = running ? "LIVE / PROCESSING" : "OFFLINE";
  $("feedPlaceholder").style.display = running ? "none" : "flex";
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok || !data.ok) throw new Error(data.error || "Request failed.");
  return data;
}

async function startCamera() {
  try {
    await postJson("/api/start-camera", {camera_id: 0});
    setRunning(true);
    refreshStatus();
  } catch (error) {
    alert(error.message);
  }
}

async function startVideo() {
  const file = $("videoFile").files[0];
  if (!file) {
    alert("Choose a test video first.");
    return;
  }
  const form = new FormData();
  form.append("video", file);
  try {
    const response = await fetch("/api/start-video", {method: "POST", body: form});
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Could not start video.");
    setRunning(true);
    refreshStatus();
  } catch (error) {
    alert(error.message);
  }
}

async function stopMonitoring() {
  try {
    await postJson("/api/stop");
    setRunning(false);
  } catch (error) {
    alert(error.message);
  }
}

function renderEvents(events) {
  const root = $("events");
  if (!events || !events.length) {
    root.innerHTML = '<p class="muted">No events yet.</p>';
    return;
  }
  root.innerHTML = events.slice(0, 30).map(e =>
    `<div class="event ${e.level || ""}"><time>${escapeHtml(e.time)}</time>${escapeHtml(e.message)}</div>`
  ).join("");
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    $("topStatus").textContent = data.error ? "MODEL / SYSTEM ERROR" : (data.running ? "MONITORING" : "SYSTEM READY");
    $("status").textContent = data.status || "READY";
    $("status").className = data.status && data.status.includes("VIOLENCE") ? "" : "status-ready";
    $("people").textContent = data.people ?? "—";
    $("confidence").textContent = Number(data.confidence || 0).toFixed(2);
    $("flow").textContent = Number(data.flow || 0).toFixed(2);
    $("source").textContent = data.running ? `Source: ${data.source}` : (data.error || "No source selected");
    renderEvents(data.events);
    if (data.running) {
      setRunning(true);
      $("feedPlaceholder").style.display = "none";
    } else {
      setRunning(false);
    }
  } catch (e) {
    $("topStatus").textContent = "BACKEND DISCONNECTED";
  }
}

$("startCamera").addEventListener("click", startCamera);
$("startVideo").addEventListener("click", startVideo);
$("stop").addEventListener("click", stopMonitoring);
$("cameraFeed").addEventListener("load", () => {
  if ($("feedBadge").textContent !== "OFFLINE") $("feedPlaceholder").style.display = "none";
});

$("visitorForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const name = $("visitorName").value.trim();
  const purpose = $("visitorPurpose").value.trim();
  const decision = $("visitorDecision").value;
  if (!name || !purpose) return;
  const tbody = $("visitorRows");
  if (tbody.querySelector("td[colspan]")) tbody.innerHTML = "";
  const row = document.createElement("tr");
  const values = [new Date().toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"}), name, purpose];
  values.forEach(value => {
    const td = document.createElement("td");
    td.textContent = value;
    row.appendChild(td);
  });
  const td = document.createElement("td");
  const badge = document.createElement("span");
  badge.className = "decision " + (decision === "Authorised" ? "authorised" : decision === "Denied" ? "denied" : "");
  badge.textContent = decision;
  td.appendChild(badge);
  row.appendChild(td);
  tbody.prepend(row);
  $("visitorName").value = "";
  $("visitorPurpose").value = "";
});

refreshStatus();
pollTimer = setInterval(refreshStatus, 1000);
