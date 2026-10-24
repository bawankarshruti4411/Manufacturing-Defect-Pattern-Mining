/* All model logic lives in Flask. This file only calls the API and draws the results. */
const $ = (s) => document.querySelector(s);
const COL = { ok: "#2f7d67", warn: "#b67730", bad: "#a95a4e", cy: "#2d3d4f" };
const GROUPS = {
  Production: ["ProductionVolume", "ProductionCost", "InventoryTurnover", "StockoutRate"],
  Quality: ["SupplierQuality", "DeliveryDelay", "QualityScore"],
  Maintenance: ["MaintenanceHours", "DowntimePercentage"],
  Workforce: ["WorkerProductivity", "SafetyIncidents"],
  Energy: ["EnergyConsumption", "EnergyEfficiency"],
  Materials: ["AdditiveProcessTime", "AdditiveMaterialCost"],
};
let SUM = null, INS = null, batchData = null;

const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const fmt = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : Number.isInteger(v) ? v : (+v).toFixed(Math.abs(v) < 1 ? 3 : 2));
const nice = (s) => s.replace(/([a-z])([A-Z])/g, "$1 $2");
const riskColor = (r) => (r === "High" ? COL.bad : r === "Moderate" ? COL.warn : COL.ok);

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

/* ---------- navigation ---------- */
function show(page) {
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === page));
  document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (page === "insights") loadInsights();
}
document.querySelectorAll("#nav button").forEach((b) => (b.onclick = () => show(b.dataset.page)));
document.querySelectorAll("[data-go]").forEach((b) => (b.onclick = () => show(b.dataset.go)));

/* ---------- small chart helpers (pure SVG / CSS) ---------- */
function donut(el, legendEl, parts) {
  const total = parts.reduce((a, p) => a + p.v, 0) || 1, R = 70, C = 2 * Math.PI * R;
  let off = 0, arcs = "";
  parts.forEach((p) => {
    const len = (p.v / total) * C;
    arcs += `<circle r="${R}" cx="100" cy="100" fill="none" stroke="${p.c}" stroke-width="26" stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-off}" transform="rotate(-90 100 100)"/>`;
    off += len;
  });
  $(el).innerHTML = `<svg viewBox="0 0 200 200" style="width:200px;display:block;margin:auto">${arcs}
    <text x="100" y="98" text-anchor="middle" fill="#e6eefc" font-size="26" font-weight="700" font-family="monospace">${pct(parts[0].v / total, 0)}</text>
    <text x="100" y="118" text-anchor="middle" fill="#8aa0c4" font-size="10">${parts[0].n.toUpperCase()}</text></svg>`;
  $(legendEl).innerHTML = parts.map((p) => `<span><i style="background:${p.c}"></i>${p.n} · ${p.v.toLocaleString()}</span>`).join("");
}
function bars(el, rows, { max, color, valFmt = (v) => v.toFixed(3) } = {}) {
  const m = max ?? Math.max(...rows.map((r) => Math.abs(r.v)), 1e-9);
  $(el).innerHTML = rows.map((r) => `<div class="bar"><span class="n" title="${r.n}">${r.n}</span><div class="t"><div class="f" data-w="${(Math.abs(r.v) / m) * 100}" style="${r.c || color ? `--c:${r.c || color}` : ""}"></div></div><span class="v">${valFmt(r.v)}</span></div>`).join("");
  requestAnimationFrame(() => setTimeout(() => document.querySelectorAll(`${el} .f`).forEach((f) => (f.style.width = f.dataset.w + "%")), 30));
}
const kpi = (label, value, sub = "", c = "") => `<div class="kpi" ${c ? `style="--c:${c}"` : ""}><span>${label}</span><b>${value}</b><small>${sub}</small></div>`;

/* ---------- dashboard ---------- */
async function init() {
  try {
    await api("/api/health");
    $("#status").className = "status ok"; $("#status span").textContent = "API online · model loaded";
    SUM = await api("/api/summary");
  } catch (e) {
    $("#status").className = "status down"; $("#status span").textContent = "API offline";
    $("#kpis").innerHTML = kpi("Backend not reachable", "—", "Start it with: python backend/app.py", COL.bad);
    return;
  }
  const d = SUM.dataset, m = SUM.metrics;
  $("#kpis").innerHTML = kpi("Total records", d.records.toLocaleString(), `${SUM.features} model features`) +
    kpi("Defective share", pct(d.defective_pct), `${d.defective.toLocaleString()} records`, COL.bad) +
    kpi("Final model", SUM.model, "selected by ROC-AUC", COL.cy) +
    kpi("ROC-AUC", m.roc_auc.toFixed(3), `F1 ${m.f1.toFixed(3)} · Recall ${m.recall.toFixed(3)}`, COL.ok);
  donut("#donut", "#donutLegend", [{ n: "Defective", v: d.defective, c: COL.bad }, { n: "Non-defective", v: d.non_defective, c: COL.ok }]);
  $("#pipeline").innerHTML = `<b>How it works.</b> The dataset has ${d.missing} missing values and ${d.duplicates} duplicates. It was split ${d.train_size.toLocaleString()} / ${d.test_size.toLocaleString()} (stratified), four models were trained and compared, and the winner was saved with its metadata.
    <div class="pipe"><span>Load</span><span>Clean check</span><span>EDA</span><span>Stratified split</span><span>4 models</span><span>Evaluate</span><span>Save .joblib</span><span>Flask API</span><span>This dashboard</span></div>`;
  buildForm();
  $("#requiredCols").textContent = "Required columns: " + SUM.feature_names.join(", ") + "  (DefectStatus is optional)";
}

/* ---------- predict ---------- */
function buildForm() {
  const placed = new Set(Object.values(GROUPS).flat());
  const groups = { ...GROUPS, Other: SUM.feature_names.filter((f) => !placed.has(f)) };
  $("#formGroups").innerHTML = Object.entries(groups).map(([g, fs]) => {
    const items = fs.filter((f) => SUM.feature_names.includes(f));
    if (!items.length) return "";
    return `<fieldset><legend>${g}</legend><div class="fields">${items.map((f) => {
      const s = SUM.input_stats[f];
      return `<div class="field"><label for="f_${f}">${nice(f)}<span class="rng">${fmt(s.min)}–${fmt(s.max)}</span></label>
      <input type="number" id="f_${f}" name="${f}" min="${s.min}" max="${s.max}" step="${s.is_int ? 1 : "any"}" placeholder="${fmt(s.median)}" title="Typical range seen in training data: ${fmt(s.min)} to ${fmt(s.max)}"></div>`;
    }).join("")}</div></fieldset>`;
  }).join("");
}
function fill(overrides) {
  SUM.feature_names.forEach((f) => {
    const s = SUM.input_stats[f]; let v = overrides && f in overrides ? overrides[f] : s.median;
    v = Math.min(Math.max(v, s.min), s.max); $("#f_" + f).value = s.is_int ? Math.round(v) : +v.toFixed(2);
    $("#f_" + f).classList.remove("bad");
  });
  $("#predictErr").textContent = "";
}
$("#fillMedian").onclick = () => fill();
$("#fillLow").onclick = () => fill({ MaintenanceHours: 4, QualityScore: 92, ProductionVolume: 300 });
$("#fillHigh").onclick = () => fill({ MaintenanceHours: 16, QualityScore: 68, ProductionVolume: 700 });

function gauge(p) {
  const c = riskColor(p < 0.3 ? "Low" : p <= 0.7 ? "Moderate" : "High"), a = Math.PI * (1 - p);
  const x = 100 + 80 * Math.cos(a), y = 100 - 80 * Math.sin(a);
  return `<svg class="gauge" viewBox="0 0 200 118"><path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#ffffff12" stroke-width="16" stroke-linecap="round"/>
  <path d="M20 100 A80 80 0 0 1 ${x.toFixed(1)} ${y.toFixed(1)}" fill="none" stroke="${c}" stroke-width="16" stroke-linecap="round"/>
  <text x="100" y="94" text-anchor="middle" fill="#161616" font-size="30" font-weight="700" font-family="monospace">${pct(p)}</text>
  <text x="100" y="112" text-anchor="middle" fill="#8aa0c4" font-size="9" letter-spacing="2">DEFECT PROBABILITY</text></svg>`;
}
$("#analyzeBtn").onclick = async () => {
  const btn = $("#analyzeBtn"), body = {}; let bad = false;
  SUM.feature_names.forEach((f) => {
    const el = $("#f_" + f), v = el.value.trim(); const ok = v !== "" && isFinite(+v);
    el.classList.toggle("bad", !ok); if (!ok) bad = true; else body[f] = +v;
  });
  if (bad) { $("#predictErr").textContent = "Please fill every field with a valid number (or use “Fill with typical values”)."; return; }
  $("#predictErr").textContent = ""; btn.disabled = true; btn.textContent = "Analyzing…";
  try {
    const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const c = r.prediction ? COL.bad : COL.ok, rc = riskColor(r.risk_level);
    const note = { High: "This record resembles historical records that were labelled defective.", Moderate: "The model is uncertain — this record sits between the two patterns.", Low: "This record resembles historical non-defective records." }[r.risk_level];
    $("#resultCard").innerHTML = `<div class="verdict" style="--c:${c}"><div class="tag">PREDICTION</div><div class="big">${r.label.toUpperCase()}</div><span class="pill" style="--c:${rc}">${r.risk_level.toUpperCase()} RISK</span></div>
      ${gauge(r.probability)}<p class="interp">${note}</p><h3>Factors moving this prediction <small>(importance-weighted)</small></h3><div id="contrib"></div>
      <p class="note" style="font-size:11px;margin-top:10px">Red = this value is closer to the average defective record; green = closer to the average non-defective record. These are associations in the data, not proven causes.</p>`;
    bars("#contrib", r.contributions.map((x) => ({ n: nice(x.feature), v: x.effect, c: x.effect > 0 ? COL.bad : COL.ok })), { max: Math.max(...r.contributions.map((x) => Math.abs(x.effect)), 1e-9), valFmt: (v) => "" });
    r.contributions.forEach((x, i) => { const el = document.querySelectorAll("#contrib .v")[i]; if (el) { el.textContent = x.leans === "defective" ? "▲ def." : "▼ ok"; el.style.color = x.leans === "defective" ? COL.bad : COL.ok; } });
  } catch (e) { $("#predictErr").textContent = e.message; }
  btn.disabled = false; btn.textContent = "Analyze Production";
};

/* ---------- batch ---------- */
const drop = $("#drop");
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && runBatch(e.dataTransfer.files[0]));
$("#file").onchange = (e) => e.target.files[0] && runBatch(e.target.files[0]);

async function runBatch(file) {
  $("#batchErr").textContent = ""; $("#batchOut").hidden = true; $("#fname").textContent = "Analyzing " + file.name + "…";
  const fd = new FormData(); fd.append("file", file);
  try {
    const r = await api("/api/batch-predict", { method: "POST", body: fd });
    batchData = r; $("#fname").textContent = "✓ " + file.name; $("#batchOut").hidden = false;
    $("#batchKpis").innerHTML = kpi("Total records", r.total.toLocaleString()) + kpi("Predicted defective", r.defective.toLocaleString(), pct(r.defective / r.total), COL.bad) +
      kpi("Predicted non-defective", r.non_defective.toLocaleString(), pct(r.non_defective / r.total), COL.ok) + kpi("Avg defect probability", pct(r.avg_probability), "", COL.warn) +
      (r.accuracy_vs_actual != null ? kpi("Match with DefectStatus", pct(r.accuracy_vs_actual), "uploaded labels", COL.cy) : "");
    bars("#riskBars", ["Low", "Moderate", "High"].map((k) => ({ n: k + " risk", v: r.risk_counts[k], c: riskColor(k) })), { max: r.total, valFmt: (v) => v.toLocaleString() });
    donut("#batchDonut", "#batchLegend", [{ n: "Defective", v: r.defective, c: COL.bad }, { n: "Non-defective", v: r.non_defective, c: COL.ok }]);
    const shown = r.rows.slice(0, 100);
    $("#rowNote").textContent = `(showing ${shown.length} of ${r.total})`;
    $("#resTable").innerHTML = "<thead><tr><th>#</th>" + r.columns.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>" +
      shown.map((row, i) => `<tr><td>${i + 1}</td>` + r.columns.map((c) => {
        const v = row[c]; if (c === "PredictionLabel") return `<td><span class="chip ${v}">${v}</span></td>`;
        if (c === "RiskLevel") return `<td><span class="chip ${v}">${v}</span></td>`;
        if (c === "DefectProbability") return `<td>${pct(v)}</td>`; return `<td>${typeof v === "number" ? fmt(v) : v}</td>`;
      }).join("") + "</tr>").join("") + "</tbody>";
  } catch (e) { $("#batchErr").textContent = "⚠ " + e.message; $("#fname").textContent = ""; }
  $("#file").value = "";
}
$("#dlBtn").onclick = () => {
  if (!batchData) return;
  const esc = (v) => { v = String(v ?? ""); return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v; };
  const csv = [batchData.columns.join(",")].concat(batchData.rows.map((r) => batchData.columns.map((c) => esc(r[c])).join(","))).join("\n");
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: "defect_predictions.csv" });
  document.body.appendChild(a); a.click(); a.remove();
};
$("#sampleBtn").onclick = () => {
  if (!SUM) return; let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const rows = Array.from({ length: 12 }, () => SUM.feature_names.map((f) => { const s = SUM.input_stats[f], v = s.min + rnd() * (s.max - s.min); return s.is_int ? Math.round(v) : +v.toFixed(3); }).join(","));
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([SUM.feature_names.join(",") + "\n" + rows.join("\n")], { type: "text/csv" })), download: "sample_batch.csv" });
  document.body.appendChild(a); a.click(); a.remove();
};

/* ---------- insights ---------- */
async function loadInsights() {
  if (INS) return;
  try { INS = await api("/api/insights"); } catch (e) { $("#metricKpis").innerHTML = kpi("Could not load insights", "—", e.message, COL.bad); return; }
  const m = INS.metrics;
  const IMP = Object.entries(INS.feature_importance).sort((a, b) => b[1] - a[1]); // API keys arrive alphabetically
  $("#metricKpis").innerHTML = [["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall], ["F1-score", m.f1], ["ROC-AUC", m.roc_auc]].map(([k, v]) => kpi(k, v.toFixed(3), INS.model_name)).join("");
  $("#compare").innerHTML = '<div class="dual" id="cmpA"></div>';
  const rows = [];
  INS.comparison.forEach((r) => { rows.push({ n: r.model + " · AUC", v: r.roc_auc, c: r.model === INS.model_name ? COL.cy : "#3b82f6" }); rows.push({ n: r.model + " · F1", v: r.f1, c: "#8aa0c4" }); });
  bars("#cmpA", rows, { max: 1 });
  const c = INS.confusion; $("#cmName").textContent = "(" + INS.model_name + ", test set)";
  $("#cm").innerHTML = `<div class="cm"><span></span><span class="h">Pred. non-defective</span><span class="h">Pred. defective</span>
    <span class="h">Actual<br>non-def.</span><div class="c" style="background:#34d39922;color:${COL.ok}">${c.tn}<small>true negative</small></div><div class="c" style="background:#f8717114;color:${COL.bad}">${c.fp}<small>false positive</small></div>
    <span class="h">Actual<br>defective</span><div class="c" style="background:#f8717114;color:${COL.bad}">${c.fn}<small>false negative</small></div><div class="c" style="background:#34d39922;color:${COL.ok}">${c.tp}<small>true positive</small></div></div>`;
  bars("#importance", IMP.slice(0, 10).map(([k, v]) => ({ n: nice(k), v: Math.max(v, 0) })), { valFmt: (v) => v.toFixed(4) });
  const g = INS.group_analysis;
  $("#groupTable").innerHTML = "<thead><tr><th>Feature</th><th>Non-defective</th><th>Defective</th><th>Corr.</th></tr></thead><tbody>" +
    Object.entries(g).sort((a, b) => Math.abs(b[1].corr) - Math.abs(a[1].corr)).map(([k, v]) => `<tr><td>${nice(k)}${INS.excluded_features.includes(k) ? " *" : ""}</td><td>${fmt(v.non_defective)}</td><td class="${v.defective > v.non_defective ? "up" : "down"}">${fmt(v.defective)}</td><td>${v.corr.toFixed(3)}</td></tr>`).join("") + "</tbody>";
  const top = IMP.slice(0, 3).map((e) => nice(e[0])).join(", ");
  $("#caveat").innerHTML = `<b>Top predictive signals:</b> ${top}. Higher or lower values of these are <i>associated</i> with higher predicted defect risk in this dataset — this is not proof of cause.<br>
    <b>Data-leakage decision:</b> * ${INS.leakage_note}<br><b>Class imbalance:</b> only ${pct(1 - INS.dataset.defective_pct)} of records are non-defective, so accuracy alone is misleading; precision, recall, F1 and ROC-AUC are reported and the model was selected on ROC-AUC.`;
}
init();
