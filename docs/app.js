"use strict";
// Website logic: loads the JSON files written by src/export_site.py and fills every number,
// table, chart, the value explorer and the price estimator. No number is typed into the page.

const REPO_URL = "https://github.com/roddersom-1209/ES-335-Project";

const DATA = {};
const FILES = {
  results: "results.json", model: "model.json", phones: "phones.json",
  premiums: "brand_premiums.json", specs: "spec_values.json", ols: "ols.json", pca: "pca.json", boosting: "boosting.json",
};
const BANDS = [[0, 15000, "< ₹15k"], [15000, 30000, "₹15–30k"], [30000, 60000, "₹30–60k"], [60000, Infinity, "₹60k+"]];
const CHIP_LABELS = {
  flagship_sd8_or_apple: "Flagship (Snapdragon 8 or Apple A-series)", snapdragon_other: "Other Snapdragon",
  dimensity_high: "Dimensity 8000/9000", dimensity_other: "Other Dimensity", helio: "MediaTek Helio",
  exynos: "Samsung Exynos", tensor: "Google Tensor", kirin: "Huawei Kirin", unisoc: "Unisoc", other: "Other chip",
};
const INPUT_INFO = {   // label, unit, step for the estimator and the variables table
  ram_gb: ["RAM", "GB", 1], storage_gb: ["Storage", "GB", 1], battery_mah: ["Battery", "mAh", 50],
  screen_in: ["Screen size", "inches", 0.01], main_back_mp: ["Main camera", "MP", 1], front_mp: ["Front camera", "MP", 1],
  n_back_cameras: ["Back cameras", "", 1], weight_g: ["Weight", "g", 1], year: ["Launch year", "", 1],
};

// ---------------------------------------------------------------------------- formatting
const inr = v => "₹" + Math.round(v).toLocaleString("en-IN");
const FMT = {
  int: v => Math.round(v).toLocaleString("en-IN"),
  raw: v => String(v),
  r2: v => v.toFixed(3),
  pct1: v => v.toFixed(1) + "%",
  pct0: v => Math.round(v) + "%",
  signed0: v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(Math.round(v)) + "%",
  share: v => Math.round(v * 100) + "%",
  share1: v => (v * 100).toFixed(1) + "%",
  inr: inr,
};
const signedPct = v => FMT.signed0(v);

function lookup(path) {
  const [file, rest] = path.split(":");
  return rest.split(".").reduce((obj, key) => (obj == null ? undefined : obj[key]), DATA[file]);
}

function el(tag, attrs = {}, html = "") {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  node.innerHTML = html;
  return node;
}

function bandOf(price) {
  return BANDS.find(([lo, hi]) => price >= lo && price < hi)[2];
}

const narrow = () => window.innerWidth < 640;

// ---------------------------------------------------------------------------- loading
async function loadAll() {
  const entries = await Promise.all(Object.entries(FILES).map(async ([key, file]) => {
    const response = await fetch(file);
    if (!response.ok) throw new Error(`${file}: ${response.status}`);
    return [key, await response.json()];
  }));
  for (const [key, value] of entries) DATA[key] = value;
}

function fillBindings() {
  document.querySelectorAll("[data-v]").forEach(span => {
    const value = lookup(span.dataset.v);
    span.textContent = value === undefined ? "?" : FMT[span.dataset.f || "raw"](value);
  });
}

// ---------------------------------------------------------------------------- charts
function plotlyReady() {
  return typeof window.Plotly !== "undefined";
}

// On a phone, break long chart and axis titles into two lines so they fit the screen
function wrapTitle(title, maxChars = 34) {
  if (!title) return title;
  const t = typeof title === "string" ? { text: title } : Object.assign({}, title);
  if (!t.text || t.text.length <= maxChars || t.text.includes("<br>")) return t;
  const cut = t.text.lastIndexOf(" ", Math.ceil(t.text.length / 2) + 4);
  if (cut > 0) t.text = t.text.slice(0, cut) + "<br>" + t.text.slice(cut + 1);
  return t;
}

function adaptLayout(layout, container) {
  layout = Object.assign({}, layout, { autosize: true });
  layout.font = Object.assign({}, layout.font, { family: currentFont() });
  delete layout.width;
  for (const key of Object.keys(layout)) {
    if (key.startsWith("yaxis") || key.startsWith("xaxis")) layout[key] = Object.assign({}, layout[key], { automargin: true });
  }
  if (narrow()) {
    layout.margin = { l: 10, r: 10, t: 56, b: 40 };
    layout.font = Object.assign({}, layout.font, { size: 11 });
    if (layout.title) layout.title = Object.assign(wrapTitle(layout.title), { font: { size: 13 } });
    for (const key of Object.keys(layout)) {
      if ((key.startsWith("xaxis") || key.startsWith("yaxis")) && layout[key].title) layout[key].title = wrapTitle(layout[key].title, 40);
    }
    layout.legend = Object.assign({}, layout.legend, { orientation: "h", y: -0.18, x: 0, font: { size: 10 } });
  }
  if (container.classList.contains("tall")) layout.height = narrow() ? 560 : 520;
  return layout;
}

const PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] };

async function renderFigure(container) {
  const response = await fetch(container.dataset.figure);
  const fig = await response.json();
  let layout = adaptLayout(fig.layout, container);
  if (container.dataset.figure.endsWith("pred_vs_actual.json")) {
    if (layout.yaxis2) {   // the right panel shared the left panel's y-axis; give it its own log axis (avoids a Plotly warning)
      delete layout.yaxis2.matches;
      layout.yaxis2.type = "log";
      layout.yaxis2.range = layout.yaxis.range;
    }
    if (narrow()) layout = stackSubplots(layout);
  }
  await Plotly.newPlot(container, fig.data, layout, PLOT_CONFIG);
  themeChart(container);
}

// The predicted-vs-actual figure has two panels side by side; on a phone we stack them.
function stackSubplots(layout) {
  layout.xaxis = Object.assign({}, layout.xaxis, { domain: [0, 1] });
  layout.xaxis2 = Object.assign({}, layout.xaxis2, { domain: [0, 1] });
  layout.yaxis = Object.assign({}, layout.yaxis, { domain: [0.58, 1] });
  layout.yaxis2 = Object.assign({}, layout.yaxis2, { domain: [0, 0.42], title: { text: "predicted price (₹)" } });
  if (layout.annotations && layout.annotations.length === 2) {
    layout.annotations = [Object.assign({}, layout.annotations[0], { x: 0.5, y: 1, yanchor: "bottom" }),
                          Object.assign({}, layout.annotations[1], { x: 0.5, y: 0.42, yanchor: "bottom" })];
  }
  layout.height = 820;
  layout.showlegend = false;
  return layout;
}

// ---------------------------------------------------------------------------- data section tables
function fillVariablesTable() {
  const tbody = document.querySelector("#variables-table tbody");
  const meanings = {
    ram_gb: "memory", storage_gb: "storage, from the model name", battery_mah: "battery capacity",
    screen_in: "main screen diagonal", main_back_mp: "largest rear camera", front_mp: "front camera",
    n_back_cameras: "number of rear cameras", weight_g: "weight", year: "launch year",
  };
  tbody.appendChild(el("tr", {}, `<td>Price (target)</td><td>Indian launch price</td><td class="num">${inr(DATA.results.data.price_min)} – ${inr(DATA.results.data.price_max)}</td><td class="num">${inr(DATA.results.data.price_median)}</td>`));
  for (const [key, info] of Object.entries(DATA.model.inputs)) {
    const [label, unit] = INPUT_INFO[key];
    const u = unit ? " " + unit : "";
    tbody.appendChild(el("tr", {}, `<td>${label}</td><td>${meanings[key]}</td><td class="num">${info.min}${u} – ${info.max}${u}</td><td class="num">${info.median}${u}</td>`));
  }
}

function fillCleaningTable() {
  const reasons = {
    "raw file": "the CSV as downloaded from Kaggle",
    "remove tablets": "names with \"pad\" or \"tab\" are tablets, not phones",
    "drop exact duplicates": "identical rows would count a phone twice (and could land in both training and test)",
    "drop rows with missing values": "no storage size in the model name, after adding back the phones confirmed by the web lookup",
  };
  const tbody = document.querySelector("#cleaning-table tbody");
  for (const step of DATA.results.cleaning) {
    tbody.appendChild(el("tr", {}, `<td>${step.step}</td><td class="num">${FMT.int(step.rows_left)}</td><td class="num">${step.dropped ? FMT.int(step.dropped) : "–"}</td><td>${reasons[step.step] || ""}</td>`));
  }
}

// ---------------------------------------------------------------------------- results
function fillScoresTable() {
  const models = DATA.results.models;
  const best = DATA.results.best_model;
  const order = Object.keys(models).sort((a, b) => models[a].test_median_pct - models[b].test_median_pct);
  const tbody = document.querySelector("#scores-table tbody");
  for (const name of order) {
    const m = models[name];
    tbody.appendChild(el("tr", name === best ? { class: "highlight" } : {}, `
      <td>${name}</td>
      <td class="num">${FMT.r2(m.test_r2)} <span class="interval">[${FMT.r2(m.test_r2_low)}–${FMT.r2(m.test_r2_high)}]</span></td>
      <td class="num">${FMT.pct1(m.test_median_pct)} <span class="interval">[${FMT.pct1(m.test_median_pct_low)}–${FMT.pct1(m.test_median_pct_high)}]</span></td>
      <td class="num">${inr(m.test_mae_rs)}</td>
      <td class="num">${FMT.r2(m.cv_r2)}</td>
      <td class="num">${FMT.pct1(m.cv_median_pct)}</td>`));
  }
}

function fillResultsText() {
  const r = DATA.results, m = r.models, best = m[r.best_model], ridge = m.Ridge;
  const brandOnly = m["Baseline 2: brand mean"], mean = m["Baseline 1: overall mean"];
  const pair = r.paired_differences[`${r.best_model} − Ridge`];
  document.getElementById("results-summary").innerHTML =
    `<strong>${r.best_model}</strong> is the most accurate model on unseen phones: R² ${FMT.r2(best.test_r2)} ` +
    `and a median error of ${FMT.pct1(best.test_median_pct)} (95% interval ${FMT.pct1(best.test_median_pct_low)}–${FMT.pct1(best.test_median_pct_high)}), ` +
    `against ${FMT.pct1(brandOnly.test_median_pct)} when only the brand is known and ${FMT.pct1(mean.test_median_pct)} when always guessing the average. ` +
    `Ridge, the interpretable linear model, reaches R² ${FMT.r2(ridge.test_r2)} and ${FMT.pct1(ridge.test_median_pct)}. ` +
    `The gap is real: ${r.best_model.toLowerCase()} has a median error ${(-pair.delta_median_pct).toFixed(1)} percentage points lower ` +
    `(95% interval ${(-pair.high).toFixed(1)} to ${(-pair.low).toFixed(1)}), and is better in ${FMT.share1(pair.share_first_better)} of bootstrap resamples. ` +
    `<strong>From cross-validation to test</strong> there was no measurable drop: ${r.best_model.toLowerCase()} went from R² ${FMT.r2(best.cv_r2)} to ${FMT.r2(best.test_r2)}, ` +
    `ridge from ${FMT.r2(ridge.cv_r2)} to ${FMT.r2(ridge.test_r2)} (its median error from ${FMT.pct1(ridge.cv_median_pct)} to ${FMT.pct1(ridge.test_median_pct)}), all within the test-set noise.`;
  const w = r.within;
  document.getElementById("within-summary").innerHTML =
    `Share of test phones priced within ±10% / ±20% / ±30% of the actual price: ` +
    `${r.best_model} ${FMT.share(w[r.best_model].within_10)} / <strong>${FMT.share(w[r.best_model].within_20)}</strong> / ${FMT.share(w[r.best_model].within_30)}.<br>` +
    `Ridge ${FMT.share(w.Ridge.within_10)} / <strong>${FMT.share(w.Ridge.within_20)}</strong> / ${FMT.share(w.Ridge.within_30)}.`;
}

// ---------------------------------------------------------------------------- predicted vs actual explorer chart
function setupExplorerChart() {
  const brands = [...new Set(DATA.phones.map(p => p.brand))].sort();
  const brandSelect = document.getElementById("pva-brand");
  brands.forEach(b => brandSelect.appendChild(el("option", { value: b }, b === "Other" ? "Other (Sony, Huawei)" : b)));
  const bandSelect = document.getElementById("pva-band");
  BANDS.forEach(([, , label]) => bandSelect.appendChild(el("option", { value: label }, label)));
  document.querySelectorAll("#pva-filters select").forEach(s => s.addEventListener("change", drawExplorerChart));
  drawExplorerChart();
}

function drawExplorerChart(highlight) {
  const key = document.getElementById("pva-model").value;
  const brand = document.getElementById("pva-brand").value;
  const band = document.getElementById("pva-band").value;
  const which = document.getElementById("pva-set").value;
  const rows = DATA.phones.filter(p => (which === "all" || p.set === "test") && (!brand || p.brand === brand) && (!band || bandOf(p.price) === band));
  const modelName = key === "boosting_price" ? "Gradient boosting" : "Ridge";

  const errors = rows.map(p => Math.abs(p[key] / p.price - 1) * 100).sort((a, b) => a - b);
  const median = errors.length ? (errors[Math.floor((errors.length - 1) / 2)] + errors[Math.ceil((errors.length - 1) / 2)]) / 2 : NaN;
  const within20 = errors.length ? errors.filter(e => e <= 20).length / errors.length : NaN;
  document.getElementById("pva-summary").textContent = rows.length
    ? `${rows.length} phones shown (${which === "test" ? "test set" : "all phones, each predicted by a model that did not see it"}). ${modelName}: median error ${FMT.pct1(median)}, ${FMT.share(within20)} within ±20%.`
    : "No phones match these filters.";

  if (!plotlyReady()) return;
  const traces = [];
  for (const b of [...new Set(rows.map(p => p.brand))].sort()) {
    const sub = rows.filter(p => p.brand === b);
    traces.push({
      type: "scatter", mode: "markers", name: b === "Other" ? "Other (Sony, Huawei)" : b,
      x: sub.map(p => p.price), y: sub.map(p => p[key]), text: sub.map(p => p.name),
      customdata: sub.map(p => 100 * (p[key] / p.price - 1)),
      hovertemplate: "%{text}<br>actual ₹%{x:,.0f}<br>predicted ₹%{y:,.0f} (%{customdata:+.0f}%)<extra></extra>",
      marker: { size: 7, opacity: 0.8 },
    });
  }
  if (highlight) {
    traces.push({ type: "scatter", mode: "markers", name: highlight.name, x: [highlight.price], y: [highlight[key]],
                  marker: { size: 16, color: "rgba(0,0,0,0)", line: { color: "#000", width: 2 } }, hoverinfo: "skip" });
  }
  const all = DATA.phones.map(p => p.price);
  const lo = Math.min(...all) * 0.7, hi = Math.max(...all) * 1.3;
  for (const [f, dash] of [[1, "solid"], [1.2, "dash"], [0.8, "dash"]]) {
    traces.push({ type: "scatter", mode: "lines", x: [lo, hi], y: [lo * f, hi * f], showlegend: false, hoverinfo: "skip",
                  line: { color: f === 1 ? "#333" : "#999", dash: dash, width: 1 } });
  }
  const layout = adaptLayout({
    template: "plotly_white", title: { text: `${modelName}: predicted vs actual` },
    xaxis: { type: "log", title: { text: "actual price (₹)" } }, yaxis: { type: "log", title: { text: "predicted price (₹)" } },
    margin: { l: 60, r: 20, t: 50, b: 50 }, legend: { font: { size: 11 } },
  }, document.getElementById("pva-chart"));
  Plotly.react("pva-chart", traces, layout, PLOT_CONFIG).then(themeChart);
}

// ---------------------------------------------------------------------------- value explorer
function setupExplorer() {
  const list = document.getElementById("phone-list");
  const byName = new Map();
  for (const p of [...DATA.phones].sort((a, b) => a.name.localeCompare(b.name))) {
    byName.set(p.name.toLowerCase(), p);
    list.appendChild(el("option", { value: p.name }));
  }
  const input = document.getElementById("phone-search");
  input.addEventListener("input", () => {
    const phone = byName.get(input.value.trim().toLowerCase());
    if (phone) showPhone(phone);
  });
}

function verdictText(pct) {
  if (pct > 10) return [`priced ${FMT.pct0(pct)} above what its listed specs and brand suggest`, "above"];
  if (pct < -10) return [`priced ${FMT.pct0(-pct)} below what its listed specs and brand suggest`, "below"];
  return ["priced close to what its listed specs and brand suggest (within ±10%)", ""];
}

function showPhone(p) {
  const [ridgeText, ridgeClass] = verdictText(p.residual_pct);
  const [gbText, gbClass] = verdictText(p.residual_pct_boosting);
  const premiumPct = 100 * (p.brand_price / p.hardware_value - 1);
  const card = document.getElementById("phone-card");
  card.hidden = false;
  card.innerHTML = `
    <h3>${p.name}</h3>
    <p class="small">${p.ram_gb} GB RAM · ${p.storage_gb} GB · ${CHIP_LABELS[p.chip_group] || p.chip_group} · ${p.battery_mah} mAh · ${p.screen_in}" · ${p.weight_g} g · ${p.year} ·
      ${p.set === "test" ? "test phone (predicted by models fitted without it)" : "training phone (out-of-fold prediction: the model that predicted it did not see it)"}</p>
    <div class="figures">
      <div class="figure"><div class="label">actual launch price</div><div class="value">${inr(p.price)}</div></div>
      <div class="figure"><div class="label">hardware value (specs + chip)</div><div class="value">${inr(p.hardware_value)}</div></div>
      <div class="figure"><div class="label">typical for ${p.brand} (ridge)</div><div class="value">${inr(p.brand_price)}</div></div>
      <div class="figure"><div class="label">gradient boosting</div><div class="value">${inr(p.boosting_price)}</div></div>
    </div>
    <p>Brand premium for this phone: ${p.brand_premium_rs >= 0 ? "+" : "−"}${inr(Math.abs(p.brand_premium_rs))} (${signedPct(premiumPct)} over its hardware value).</p>
    <p>Ridge: <span class="verdict ${ridgeClass}">${ridgeText}</span> (actual vs ${inr(p.brand_price)}, ${signedPct(p.residual_pct)}).<br>
       Gradient boosting: <span class="verdict ${gbClass}">${gbText}</span> (${signedPct(p.residual_pct_boosting)}).</p>
    ${p.india_price_flag ? `<p class="note warn">Doubtful India price: ${p.india_price_flag}.</p>` : ""}
    <p class="small">"Above" or "below" is not the same as overpriced or a bargain: camera quality, display, materials and software support are not in our data.</p>`;
  if (plotlyReady()) drawExplorerChart(p.set === "test" || document.getElementById("pva-set").value === "all" ? p : null);
}

// ---------------------------------------------------------------------------- estimator
function setupEstimator() {
  const model = DATA.model, form = document.getElementById("estimator-form");
  for (const [key, info] of Object.entries(model.inputs)) {
    const [label, unit, step] = INPUT_INFO[key];
    const u = unit ? " " + unit : "";
    let field;
    if (info.values) {
      field = el("select", { id: "in-" + key });
      const nearest = info.values.reduce((a, b) => Math.abs(b - info.median) < Math.abs(a - info.median) ? b : a);
      info.values.forEach(v => field.appendChild(el("option", v === nearest ? { value: v, selected: "" } : { value: v }, v + u)));
    } else {
      field = el("input", { id: "in-" + key, type: "number", min: info.min, max: info.max, step: step, value: info.median, inputmode: "decimal" });
    }
    const wrap = el("label", {}, `${label}${unit ? " (" + unit + ")" : ""}`);
    wrap.appendChild(field);
    if (!info.values) wrap.appendChild(el("span", { class: "range" }, `range ${info.min}–${info.max}`));
    form.appendChild(wrap);
  }
  const chip = el("select", { id: "in-chip_group" });
  model.chip_groups.forEach(c => chip.appendChild(el("option", c === "snapdragon_other" ? { value: c, selected: "" } : { value: c }, CHIP_LABELS[c] || c)));
  const brand = el("select", { id: "in-brand" });
  model.brands.forEach(b => brand.appendChild(el("option", b === "Samsung" ? { value: b, selected: "" } : { value: b }, b === "Other" ? "Other (Sony, Huawei)" : b)));
  for (const [label, field] of [["Chip group", chip], ["Brand", brand]]) {
    const wrap = el("label", {}, label);
    wrap.appendChild(field);
    form.appendChild(wrap);
  }
  form.addEventListener("input", runEstimator);
  form.addEventListener("submit", e => e.preventDefault());

  const r = DATA.results.models;
  document.getElementById("estimator-accuracy").innerHTML =
    `<br>Typical error about ${FMT.pct0(r.Ridge.test_median_pct)} (ridge, median on unseen phones).<br>Our most accurate model, gradient boosting, about ${FMT.pct0(r["Gradient boosting"].test_median_pct)}.`;
  runEstimator();
}

function predict(part, x, cats) {
  let total = part.intercept;
  part.numeric_features.forEach((name, j) => {
    total += part.means ? part.numeric_coefs[j] * (x[name] - part.means[j]) / part.scales[j] : part.numeric_coefs[j] * x[name];
  });
  for (const [col, value] of Object.entries(cats)) {
    if (part.categories[col]) total += part.categories[col][value] || 0;   // unseen category -> 0
  }
  return total;
}

// Gradient boosting from docs/boosting.json: walk each of its trees and add up the leaf values.
// sklearn compares its inputs as 32-bit floats, so we do the same with Math.fround.
function predictBoosting(x, chip, brand) {
  const b = DATA.boosting;
  const row = b.numeric_features.map(name => x[name]);
  for (const [col, value] of [["chip_group", chip], ["brand", brand]]) {
    for (const category of b.categories[col]) row.push(category === value ? 1 : 0);   // unseen -> all zeros
  }
  let total = b.init;
  for (const tree of b.trees) {
    let node = 0;
    while (tree.left[node] !== -1) {
      node = Math.fround(row[tree.feature[node]]) <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    }
    total += b.learning_rate * tree.value[node];
  }
  return total;
}

function runEstimator() {
  const model = DATA.model, raw = {}, warnings = [];
  for (const [key, info] of Object.entries(model.inputs)) {
    let v = parseFloat(document.getElementById("in-" + key).value);
    if (Number.isNaN(v)) v = info.median;
    if (v < info.min || v > info.max) {
      warnings.push(`${INPUT_INFO[key][0]} must be between ${info.min} and ${info.max}.<br>We used the nearest allowed value.`);
      v = Math.min(info.max, Math.max(info.min, v));
    }
    raw[key] = v;
  }
  const chip = document.getElementById("in-chip_group").value, brand = document.getElementById("in-brand").value;
  if (!model.observed.ram_storage.some(([r, s]) => r === raw.ram_gb && s === raw.storage_gb)) {
    warnings.push(`No training phone has ${raw.ram_gb} GB RAM with ${raw.storage_gb} GB storage, so this estimate is an extrapolation.`);
  }
  if (!model.observed.brand_chip.some(([b, c]) => b === brand && c === chip)) {
    warnings.push(`No ${brand} phone in the training data uses a "${CHIP_LABELS[chip] || chip}" chip, so this combination is unusual.`);
  }
  const x = {};
  for (const [feature, t] of Object.entries(model.transforms)) {
    x[feature] = t.transform === "log2" ? Math.log2(raw[t.from]) : raw[t.from];
  }
  const hardware = Math.exp(predict(model.model_a, x, { chip_group: chip }));
  const brandPrice = Math.exp(predict(model.ridge, x, { chip_group: chip, brand: brand }));
  const premium = brandPrice - hardware, premiumPct = 100 * (brandPrice / hardware - 1);
  const boostingPrice = Math.exp(predictBoosting(x, chip, brand));
  const r = DATA.results.models;

  const warn = document.getElementById("estimator-warning");
  warn.hidden = warnings.length === 0;
  warn.innerHTML = warnings.map(w => `<div>${w}</div>`).join("");
  const brandName = brand === "Other" ? "Other (Sony, Huawei)" : brand;
  document.getElementById("estimator-output").innerHTML = `
    <div class="figures two">
      <div class="figure main"><div class="label">Ridge: typical price for ${brandName}</div><div class="value">${inr(brandPrice)}</div>
        <div class="label">typical error about ${FMT.pct0(r.Ridge.test_median_pct)}</div></div>
      <div class="figure main"><div class="label">Gradient boosting: typical price for ${brandName}</div><div class="value">${inr(boostingPrice)}</div>
        <div class="label">typical error about ${FMT.pct0(r["Gradient boosting"].test_median_pct)} (our most accurate model)</div></div>
    </div>
    <div class="figures three">
      <div class="figure"><div class="label">hardware value (specs + chip, Model A)</div><div class="value">${inr(hardware)}</div></div>
      <div class="figure"><div class="label">brand premium (ridge price − hardware value)</div><div class="value">${premium >= 0 ? "+" : "−"}${inr(Math.abs(premium))}</div></div>
      <div class="figure"><div class="label">brand premium (%)</div><div class="value">${signedPct(premiumPct)}</div></div>
    </div>
    <p class="small">Estimates of typical launch pricing in this dataset, not a phone's true worth.<br>
      The hardware value and the brand premium come from two different linear models (with and without brand), so the "premium" also contains small differences between those models.<br>
      Gradient boosting gives only a price, not a split into hardware and brand. It changes in steps, and it cannot go beyond the prices of the training phones.</p>`;
}

// ---------------------------------------------------------------------------- premiums and spec values
function fillPremiums() {
  const ul = document.getElementById("premium-caveats");
  for (const p of DATA.premiums.filter(p => p.caveat)) {
    ul.appendChild(el("li", {}, `<strong>${p.label}: ${signedPct(p.premium_pct)}</strong> <span class="interval">[${signedPct(p.low_pct)} to ${signedPct(p.high_pct)}]</span> — ${p.caveat.replace(/\. /g, ".<br>")}`));
  }
  const r = DATA.results.models;
  document.getElementById("premium-accuracy").innerHTML =
    `These numbers come from ridge, which prices unseen phones with R² ${FMT.r2(r.Ridge.test_r2)} and a median error of ${FMT.pct1(r.Ridge.test_median_pct)}.<br>` +
    `The most accurate model, gradient boosting, reaches R² ${FMT.r2(r["Gradient boosting"].test_r2)} and ${FMT.pct1(r["Gradient boosting"].test_median_pct)}.`;
  const tbody = document.querySelector("#spec-table tbody");
  for (const s of DATA.specs) {
    tbody.appendChild(el("tr", {}, `<td>${s.change}</td><td class="num">${signedPct(s.pct)}</td><td class="num">${signedPct(s.low_pct)} to ${signedPct(s.high_pct)}</td><td>${s.note.replace(/\. /g, ".<br>")}</td>`));
  }
}

// ---------------------------------------------------------------------------- OLS section
const pValue = p => (p < 0.001 ? "&lt;0.001" : p.toFixed(3));
const withInterval = r => `${signedPct(r.pct)} <span class="interval">[${signedPct(r.low_pct)} to ${signedPct(r.high_pct)}]</span>`;

function fillOls() {
  const ols = DATA.ols, m = DATA.results.models;
  const scores = document.querySelector("#ols-scores tbody");
  for (const name of ["Model A: specs + chip", "Model B: specs + chip + brand", "Ridge"]) {
    const s = m[name];
    scores.appendChild(el("tr", {}, `<td>${name}${name === "Ridge" ? " (for comparison)" : ""}</td><td class="num">${FMT.r2(s.cv_r2)}</td><td class="num">${FMT.pct1(s.cv_median_pct)}</td><td class="num">${FMT.r2(s.test_r2)}</td><td class="num">${FMT.pct1(s.test_median_pct)}</td>`));
  }
  const a = Object.fromEntries(ols.model_a.map(r => [r.term, r]));
  const b = Object.fromEntries(ols.model_b.map(r => [r.term, r]));
  const specs = document.querySelector("#ols-specs tbody");
  for (const r of ols.model_b.filter(r => r.change !== "vs reference")) {
    specs.appendChild(el("tr", {}, `<td>${r.change}</td><td class="num">${withInterval(a[r.term])}</td><td class="num">${withInterval(r)}</td><td class="num">${pValue(r.p_value)}</td>`));
  }
  const both = t => 100 * ((1 + t.log2_ram.pct / 100) * (1 + t.log2_storage.pct / 100) - 1);
  document.getElementById("ols-specs-text").innerHTML =
    `Model A says doubling RAM goes with ${signedPct(a.log2_ram.pct)} and doubling storage with ${signedPct(a.log2_storage.pct)}.<br>` +
    `Model B says ${signedPct(b.log2_ram.pct)} and ${signedPct(b.log2_storage.pct)}. The split moves because RAM and storage rise together.<br>` +
    `But doubling <em>both</em> is stable: ${signedPct(both(a))} in Model A and ${signedPct(both(b))} in Model B. ` +
    `A bigger battery goes with a lower price (${signedPct(b.battery_mah.pct)} per +1000 mAh in Model B): at equal specs, large batteries mark budget phones. This is an association, not a cause.`;
  const chips = document.querySelector("#ols-chips tbody");
  for (const r of ols.model_b.filter(r => r.term.startsWith("chip: ")).sort((x, y) => y.pct - x.pct)) {
    const key = r.term.slice(6);
    chips.appendChild(el("tr", {}, `<td>${CHIP_LABELS[key] || key}</td><td class="num">${withInterval(a[r.term])}</td><td class="num">${withInterval(r)}</td>`));
  }
  const brands = document.querySelector("#ols-brands tbody");
  for (const r of ols.model_b.filter(r => r.term.startsWith("brand: ")).sort((x, y) => y.pct - x.pct)) {
    const name = r.term.slice(7);
    brands.appendChild(el("tr", {}, `<td>${name === "Other" ? "Other (Sony, Huawei)" : name}</td><td class="num">${signedPct(r.pct)}</td><td class="num">${signedPct(r.low_pct)} to ${signedPct(r.high_pct)}</td><td class="num">${pValue(r.p_value)}</td>`));
  }
  const vif = document.querySelector("#ols-vif tbody");
  for (const r of ols.vif) {
    vif.appendChild(el("tr", {}, `<td>${r.feature}</td><td class="num">${r.vif_numeric_only.toFixed(2)}</td><td class="num">${r.vif_model_b.toFixed(2)}</td>`));
  }
}

// ---------------------------------------------------------------------------- where the model fails (section 5)
function fillFailures() {
  const r = DATA.results, eb = r.error_by_brand;
  const brands = document.querySelector("#brand-error-table tbody");
  for (const b of [...eb.brands].sort((x, y) => y.boosting_median_pct - x.boosting_median_pct)) {
    brands.appendChild(el("tr", {}, `<td>${b.brand === "Other" ? "Other (Sony, Huawei)" : b.brand}</td><td class="num">${b.phones}</td><td class="num">${FMT.pct1(b.ridge_median_pct)}</td><td class="num">${FMT.pct1(b.boosting_median_pct)}</td>`));
  }
  document.getElementById("brand-error-note").textContent =
    "Too few test phones to judge: " + eb.too_few.map(b => `${b.brand === "Other" ? "Other (Sony, Huawei)" : b.brand} (${b.phones})`).join(", ") + ".";
  const misses = document.querySelector("#misses-table tbody");
  for (const m of r.largest_misses) {
    misses.appendChild(el("tr", {}, `<td>${m.phone}</td><td>${CHIP_LABELS[m.chip_group] || m.chip_group}</td><td class="num">${inr(m.actual)}</td><td class="num">${inr(m.predicted)}</td><td class="num">${signedPct(m.error_pct)}</td><td class="num">${signedPct(m.ridge_error_pct)}</td>`));
  }
}

// ---------------------------------------------------------------------------- PCA (in "What drives price?")
const SPEC_LABELS = {
  log2_ram: "RAM (log2)", log2_storage: "Storage (log2)", battery_mah: "Battery", screen_in: "Screen size",
  main_back_mp: "Main camera", front_mp: "Front camera", n_back_cameras: "Back cameras", weight_g: "Weight", year: "Launch year",
};

function fillPca() {
  const { components, loadings } = DATA.pca;
  const tbody = document.querySelector("#pca-table tbody");
  const signed = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
  for (const feature of Object.keys(loadings).sort((a, b) => Math.abs(loadings[b].PC1) - Math.abs(loadings[a].PC1))) {
    const l = loadings[feature];
    tbody.appendChild(el("tr", {}, `<td>${SPEC_LABELS[feature] || feature}</td><td class="num">${signed(l.PC1)}</td><td class="num">${signed(l.PC2)}</td><td class="num">${signed(l.PC3)}</td>`));
  }
  const k90 = components.findIndex(c => c.cumulative >= 0.9) + 1;
  const [pc1, pc2] = components;
  document.getElementById("pca-text").innerHTML =
    `PC1 explains ${FMT.pct0(100 * pc1.share)} of the variation in the specs, and every spec loads positively on it: it is a "bigger, newer, more of everything" axis.<br>` +
    `PC2 (${FMT.pct0(100 * pc2.share)}) contrasts memory and cameras with battery, weight and screen size.<br>` +
    `${k90} of the ${components.length} components are needed to keep 90% of the variation, so the specs overlap only moderately.<br>` +
    `No component tracks the price on its own (PC1 correlates ${pc1.corr_log_price.toFixed(2)} with log price, PC2 ${pc2.corr_log_price.toFixed(2)}), so we keep all the specs in the models and use PCA to understand the data.`;
}

// ---------------------------------------------------------------------------- testing section
function fillTesting() {
  const r = DATA.results, m = r.models;
  const cv = document.querySelector("#cv-table tbody");
  for (const name of Object.keys(m).sort((x, y) => m[x].cv_median_pct - m[y].cv_median_pct)) {
    const s = m[name];
    cv.appendChild(el("tr", {}, `<td>${name}</td><td class="num">${FMT.r2(s.cv_r2)} ± ${FMT.r2(s.cv_r2_std)}</td><td class="num">${FMT.pct1(s.cv_median_pct)} ± ${s.cv_median_pct_std.toFixed(1)}</td><td class="num">${inr(s.cv_mae_rs)} ± ${inr(s.cv_mae_rs_std)}</td>`));
  }
  const paired = document.querySelector("#paired-table tbody");
  const clear = [], unclear = [];
  for (const [name, d] of Object.entries(r.paired_differences)) {
    const pts = v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1);
    paired.appendChild(el("tr", {}, `<td>${name}</td><td class="num">${pts(d.delta_median_pct)} points</td><td class="num">${pts(d.low)} to ${pts(d.high)}</td><td class="num">${FMT.share1(d.share_first_better)}</td><td class="num">${(d.delta_r2 >= 0 ? "+" : "−") + Math.abs(d.delta_r2).toFixed(3)}</td>`));
    (d.low > 0 || d.high < 0 ? clear : unclear).push(name);
  }
  document.getElementById("paired-text").innerHTML =
    `Clearly real (interval excludes 0): ${clear.join(", ") || "none"}.<br>` +
    `Within the noise (cannot be separated on this test set): ${unclear.join(", ") || "none"}.`;
  const robust = document.querySelector("#robust-table tbody");
  for (const [name, s] of Object.entries(r.without_flagged)) {
    robust.appendChild(el("tr", {}, `<td>${name}</td><td class="num">R² ${FMT.r2(m[name].test_r2)} · ${FMT.pct1(m[name].test_median_pct)} <span class="interval">(${r.data.test_phones} phones)</span></td><td class="num">R² ${FMT.r2(s.r2)} · ${FMT.pct1(s.median_pct)} <span class="interval">(${s.test_phones} phones)</span></td>`));
  }
}

// ---------------------------------------------------------------------------- conclusions
function fillConclusions() {
  const r = DATA.results, m = r.models, best = m[r.best_model];
  const prem = Object.fromEntries(DATA.premiums.map(p => [p.brand, p]));
  const top = DATA.premiums.reduce((a, b) => (b.premium_pct > a.premium_pct ? b : a));
  const bottom = DATA.premiums.reduce((a, b) => (b.premium_pct < a.premium_pct ? b : a));
  const factor = (1 + top.premium_pct / 100) / (1 + bottom.premium_pct / 100);
  const both = DATA.specs.find(s => s.change.startsWith("doubling BOTH"));
  const camera = DATA.specs.find(s => s.change.includes("camera"));
  document.getElementById("conclusions-text").innerHTML = `
    <p><strong>A phone's Indian launch price can be predicted fairly well from its listed specs, chip and brand.</strong>
    On unseen phones, ${r.best_model.toLowerCase()} explains R² ${FMT.r2(best.test_r2)} of the variation in log price, with a median error of ${FMT.pct1(best.test_median_pct)}
    (95% interval ${FMT.pct1(best.test_median_pct_low)}–${FMT.pct1(best.test_median_pct_high)}).<br>${FMT.share(r.within[r.best_model].within_20)} of phones are priced within ±20%.</p>
    <p><strong>Hardware explains most of the price, and brand adds a clear extra.</strong> Specs and chip alone (Model A) reach R² ${FMT.r2(m["Model A: specs + chip"].test_r2)} on the test set.<br>
    Adding brand (ridge) raises this to ${FMT.r2(m.Ridge.test_r2)}, while brand alone reaches only ${FMT.r2(m["Baseline 2: brand mean"].test_r2)}.
    For the same specs and chip group, ${top.label} phones cost about ${signedPct(top.premium_pct)} more than an average brand${top.brand === "Apple" ? " (relative to a phone with a flagship chip)" : ""},
    and ${bottom.label} phones about ${signedPct(bottom.premium_pct)}: a factor of about ${factor.toFixed(1)} between them.</p>
    <p><strong>What specs are worth:</strong> doubling RAM and storage together goes with about ${signedPct(both.pct)} (${signedPct(both.low_pct)} to ${signedPct(both.high_pct)}),
    while camera megapixels add nothing measurable (${signedPct(camera.pct)}).
    Tree models predict better than linear ones mainly because brands price their budget and flagship lines differently, which a single brand premium cannot capture.</p>
    <p class="small">These are associations in launch prices from one dataset, not causes.<br>See the limitations below.</p>`;
}

// ---------------------------------------------------------------------------- font and dark mode
function currentFont() {
  return getComputedStyle(document.body).fontFamily;
}

function isDark() {
  const chosen = document.documentElement.dataset.theme;
  return chosen ? chosen === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Colours of one chart for the current theme (applied after every plot and when the theme changes)
function themeChart(plot) {
  if (!plot || !plot.layout) return;
  const text = cssVar("--text"), grid = cssVar("--chart-grid"), muted = cssVar("--muted");
  const update = { paper_bgcolor: cssVar("--card"), plot_bgcolor: cssVar("--chart-bg"), "font.color": text,
                   "legend.bgcolor": "rgba(0,0,0,0)", "title.font.color": text };
  for (const key of Object.keys(plot.layout)) {
    if (/^[xy]axis\d*$/.test(key)) {
      Object.assign(update, { [key + ".gridcolor"]: grid, [key + ".zerolinecolor"]: grid, [key + ".linecolor"]: grid });
    }
  }
  (plot.layout.annotations || []).forEach((a, i) => { update[`annotations[${i}].font.color`] = text; });
  Plotly.relayout(plot, update);
  const lines = plot.data.map((t, i) => (t.mode === "lines" ? i : -1)).filter(i => i >= 0);
  if (lines.length) {
    Plotly.restyle(plot, { "line.color": lines.map(i => (plot.data[i].line && plot.data[i].line.dash && plot.data[i].line.dash !== "solid" ? muted : text)) }, lines);
  }
}

function setupTheme() {
  const button = document.getElementById("theme-toggle");
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* storage blocked: follow the system */ }
  if (saved === "dark" || saved === "light") document.documentElement.dataset.theme = saved;
  const label = () => {
    button.textContent = isDark() ? "Light mode" : "Dark mode";
    button.setAttribute("aria-pressed", String(isDark()));
  };
  const rethemeAll = () => {
    label();
    if (plotlyReady()) document.querySelectorAll(".js-plotly-plot").forEach(themeChart);
  };
  button.addEventListener("click", () => {
    document.documentElement.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (e) { /* ignore */ }
    rethemeAll();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rethemeAll);
  label();
}

// ---------------------------------------------------------------------------- start
async function main() {
  setupTheme();
  try {
    await loadAll();
  } catch (error) {
    document.getElementById("load-error").hidden = false;
    console.warn("Could not load data files:", error.message);
    return;
  }
  fillBindings();
  fillVariablesTable();
  fillCleaningTable();
  fillScoresTable();
  fillResultsText();
  fillPremiums();
  fillOls();
  fillPca();
  fillFailures();
  fillTesting();
  fillConclusions();
  setupExplorer();
  setupEstimator();
  const repo = document.getElementById("repo-link");
  repo.innerHTML = REPO_URL ? `<a href="${REPO_URL}">${REPO_URL}</a>` : "the project's GitHub repository";
  if (plotlyReady()) {
    setupExplorerChart();
    await Promise.all([...document.querySelectorAll("[data-figure]")].map(renderFigure));
  } else {
    document.querySelectorAll(".chart").forEach(c => { c.textContent = "Charts need Plotly.js, which could not be loaded."; c.style.minHeight = "0"; });
  }
}

main();
