"""Export everything the website needs into docs/ (Phase 9).

Run after notebooks 01-07:  python src/export_site.py

Writes
  docs/model.json            ridge and Model A, so the browser can compute predictions
  docs/phones.json           one record per phone (actual price, hardware value, brand price, ...)
  docs/brand_premiums.json   brand premiums with 95% intervals (Phase 8)
  docs/spec_values.json      spec values with 95% intervals (Phase 8)
  docs/results.json          every number the website text quotes
  docs/figures/*.json        Plotly figures

Every number comes from the results/ files or from refitting with the saved
settings on the training set. Nothing is typed by hand.
"""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from features import NUMERIC_FEATURES
from split import load_train_test, N_FOLDS, TEST_SIZE

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
CATEGORICAL = ["chip_group", "brand"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL

# The raw inputs a website visitor types in, and how each model feature is made from them
RAW_INPUTS = ["ram_gb", "storage_gb", "battery_mah", "screen_in", "main_back_mp",
              "front_mp", "n_back_cameras", "weight_g", "year"]
TRANSFORMS = {"log2_ram": ("ram_gb", "log2"), "log2_storage": ("storage_gb", "log2")}
for name in NUMERIC_FEATURES:
    TRANSFORMS.setdefault(name, (name, "none"))

MIN_BRAND_PHONES = 5     # brands with fewer test phones are "too few to judge"
N_LARGEST_MISSES = 10
PRICE_BANDS = [(0, 15000, "< ₹15k"), (15000, 30000, "₹15–30k"), (30000, 60000, "₹30–60k"), (60000, np.inf, "₹60k+")]


def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


def rounded(x, digits=6):
    """Python float rounded for JSON (NaN -> None). Kept precise so the website's own rounding matches the notebooks."""
    return None if pd.isna(x) else round(float(x), digits)


# ----------------------------------------------------------------------------- model.json
def export_models(train):
    """Ridge (the Phase 7 model, loaded from models/ridge.joblib) and Model A refitted with NumPy OLS."""
    ridge = joblib.load(ROOT / "models" / "ridge.joblib")
    pre = ridge.named_steps["pre"]
    scaler, encoder = pre.named_transformers_["num"], pre.named_transformers_["cat"]
    coef = ridge.named_steps["model"].coef_
    n_num = len(NUMERIC_FEATURES)
    ridge_json = {
        "description": "log_price = intercept + sum(coef_j * (x_j - mean_j) / scale_j) + chip_group coef + brand coef",
        "alpha": float(ridge.named_steps["model"].alpha),
        "numeric_features": NUMERIC_FEATURES,
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        "numeric_coefs": coef[:n_num].tolist(),
        "intercept": float(ridge.named_steps["model"].intercept_),
        "categories": {},
    }
    start = n_num
    for col, cats in zip(CATEGORICAL, encoder.categories_):   # all categories kept, one coefficient each
        ridge_json["categories"][col] = {str(c): float(b) for c, b in zip(cats, coef[start:start + len(cats)])}
        start += len(cats)

    # Model A: OLS on numeric features + chip dummies (reference snapdragon_other dropped), no brand
    chips = sorted(train["chip_group"].unique())
    reference = "snapdragon_other"
    others = [c for c in chips if c != reference]
    X = np.column_stack([np.ones(len(train)), train[NUMERIC_FEATURES].to_numpy(dtype=float)]
                        + [(train["chip_group"] == c).to_numpy(dtype=float) for c in others])
    b = np.linalg.lstsq(X, train["log_price"].to_numpy(), rcond=None)[0]   # same as (XᵀX)⁻¹Xᵀy
    model_a_json = {
        "description": "log_price = intercept + sum(coef_j * x_j) + chip_group coef (reference category = 0)",
        "numeric_features": NUMERIC_FEATURES,
        "numeric_coefs": b[1:1 + n_num].tolist(),
        "intercept": float(b[0]),
        "reference_chip_group": reference,
        "categories": {"chip_group": {reference: 0.0, **{c: float(v) for c, v in zip(others, b[1 + n_num:])}}},
    }

    inputs = {}
    for col in RAW_INPUTS:
        inputs[col] = {"min": float(train[col].min()), "max": float(train[col].max()),
                       "median": float(train[col].median())}
    for col in ["ram_gb", "storage_gb", "n_back_cameras"]:
        inputs[col]["values"] = sorted(float(v) for v in train[col].unique())
    # combinations that occur in the training data, so the website can warn about unusual inputs
    observed = {
        "ram_storage": sorted({(float(r), float(s)) for r, s in zip(train["ram_gb"], train["storage_gb"])}),
        "brand_chip": sorted({(b, c) for b, c in zip(train["brand"], train["chip_group"])}),
    }

    return {
        "target": "log_price = natural log of the launch price in ₹; price = exp(log_price)",
        "unknown_category": "a brand or chip group not listed contributes 0",
        "transforms": {f: {"from": src, "transform": t} for f, (src, t) in TRANSFORMS.items()},
        "inputs": inputs,
        "observed": observed,
        "brands": sorted(train["brand"].unique()),
        "chip_groups": chips,
        "ridge": ridge_json,
        "model_a": model_a_json,
    }


# ----------------------------------------------------------------------------- boosting.json
def export_boosting():
    """The Phase 7 gradient-boosting model (models/gradient_boosting.joblib) as plain arrays,
    so the website's estimator can walk its trees in JavaScript."""
    gb = joblib.load(ROOT / "models" / "gradient_boosting.joblib")
    pre, model = gb.named_steps["pre"], gb.named_steps["model"]
    trees = []
    for tree in model.estimators_[:, 0]:
        t = tree.tree_
        trees.append({"feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                      "left": t.children_left.tolist(), "right": t.children_right.tolist(),
                      "value": t.value[:, 0, 0].tolist()})
    encoder = pre.named_transformers_["cat"]
    return {
        "description": ("log_price = init + learning_rate * sum of tree outputs. Input columns: the numeric features, "
                        "then one 0/1 column per category (unknown category = all zeros). In each tree, start at node 0 "
                        "and go left if float32(x[feature]) <= threshold, until left == -1; the leaf's value is the output."),
        "init": float(model.init_.constant_.ravel()[0]),
        "learning_rate": float(model.learning_rate),
        "numeric_features": NUMERIC_FEATURES,
        "categories": {col: [str(c) for c in cats] for col, cats in zip(CATEGORICAL, encoder.categories_)},
        "trees": trees,
    }


# ----------------------------------------------------------------------------- phones.json
def export_phones():
    v = pd.read_csv(RESULTS / "phone_values.csv", index_col="row")
    clean = pd.read_csv(ROOT / "data" / "processed" / "phones_clean.csv")   # same row numbers as phone_values
    flags = clean["india_price_flag"].fillna("")
    records = []
    for row, r in v.iterrows():
        records.append({
            "id": int(row), "name": r["phone"], "model": r["Model Name"], "brand": r["brand"],
            "chip_group": r["chip_group"], "set": r["set"],
            **{col: rounded(r[col], 2) for col in RAW_INPUTS},
            "price": rounded(r["actual price"], 0), "hardware_value": rounded(r["hardware value"], 0),
            "brand_price": rounded(r["brand price"], 0), "boosting_price": rounded(r["boosting price"], 0),
            "brand_premium_rs": rounded(r["brand premium ₹"], 0),
            "residual_pct": rounded(r["residual % (ridge)"], 1),
            "residual_pct_boosting": rounded(r["residual % (boosting)"], 1),
            "verdict": r["verdict"], "india_price_flag": flags[row]})
    return records


# ----------------------------------------------------------------------------- premiums and spec values
CAVEATS = {
    "Apple": "Relative to a phone with a flagship chip and the same specs (Apple A-series chips are grouped with flagship Snapdragon 8 chips).",
    "Google": "Partly includes chip value: the Tensor chip is used only by Google.",
    "Other": "Sony and Huawei. Partly includes chip value: the Kirin chip is used only by Huawei. Only 12 training phones.",
}


def export_premiums():
    p = pd.read_csv(RESULTS / "brand_premiums.csv")
    return [{"brand": r["brand"], "label": "Other (Sony, Huawei)" if r["brand"] == "Other" else r["brand"],
             "n_train": int(r["training phones"]), "premium_pct": rounded(r["premium vs average %"], 1),
             "low_pct": rounded(r["low %"], 1), "high_pct": rounded(r["high %"], 1),
             "includes_zero": bool(r["interval includes 0"]),
             "vs_samsung_ridge_pct": rounded(r["ridge vs Samsung %"], 1),
             "vs_samsung_ols_pct": rounded(r["OLS Model B vs Samsung %"], 1),
             "caveat": CAVEATS.get(r["brand"], "")} for _, r in p.iterrows()]


SPEC_NOTES = {
    "doubling RAM": "How the memory effect splits between RAM and storage is unstable across models. Use the combined number.",
    "doubling storage": "How the memory effect splits between RAM and storage is unstable across models. Use the combined number.",
    "doubling BOTH RAM and storage": "The robust memory number.",
    "+1000 mAh battery": "Negative because, at equal specs, big batteries mark budget phones. An association, not a cause.",
}


def export_spec_values():
    s = pd.read_csv(RESULTS / "spec_values.csv")
    return [{"change": r["change"], "pct": rounded(r["% price change"], 1), "low_pct": rounded(r["low %"], 1),
             "high_pct": rounded(r["high %"], 1), "note": SPEC_NOTES.get(r["change"], "")} for _, r in s.iterrows()]


# ----------------------------------------------------------------------------- ols.json
def export_ols():
    """Phase 4 OLS coefficient tables (Model A and B, as % price effects) and the VIF table."""
    tables = {}
    for name, file in [("model_a", "ols_model_a_coefficients.csv"), ("model_b", "ols_model_b_coefficients.csv")]:
        t = pd.read_csv(RESULTS / file)
        tables[name] = [{"term": r["term"], "change": r["change"], "pct": rounded(r["% price effect"]),
                         "low_pct": rounded(r["95% CI low %"]), "high_pct": rounded(r["95% CI high %"]),
                         "p_value": rounded(r["p-value"], 8)} for _, r in t.iterrows()]
    vif = pd.read_csv(RESULTS / "vif.csv", index_col=0)
    tables["vif"] = [{"feature": f, "vif_numeric_only": rounded(r.iloc[0]), "vif_model_b": rounded(r.iloc[1])}
                     for f, r in vif.iterrows()]
    return tables


# ----------------------------------------------------------------------------- pca.json
def export_pca():
    """PCA of the 9 standardised specs (notebook 04_regularised_simple, section 7)."""
    variance = pd.read_csv(RESULTS / "pca_variance.csv")
    loadings = pd.read_csv(RESULTS / "pca_loadings.csv", index_col="feature")
    return {
        "components": [{"name": r["component"], "variance": rounded(r["variance (eigenvalue)"]),
                        "share": rounded(r["share of variance"]), "cumulative": rounded(r["cumulative share"]),
                        "corr_log_price": rounded(r["correlation with log price"])} for _, r in variance.iterrows()],
        "loadings": {f: {c: rounded(v) for c, v in row.items()} for f, row in loadings.iterrows()},
    }


# ----------------------------------------------------------------------------- results.json
def band_of(price):
    for low, high, label in PRICE_BANDS:
        if low <= price < high:
            return label


def export_results(train, test):
    scores = pd.read_csv(RESULTS / "test_scores.csv").set_index("model")
    boot = pd.read_csv(RESULTS / "test_bootstrap.csv").set_index("name")
    cv = pd.read_csv(RESULTS / "cv_phase6.csv").set_index("model")
    preds = pd.read_csv(RESULTS / "test_predictions.csv", index_col="row")
    actual = preds["price_inr"]

    models = {}
    for m in scores.index:
        models[m] = {
            "test_r2": rounded(scores.loc[m, "R²"]), "test_r2_low": rounded(boot.loc[m, "R² low"]),
            "test_r2_high": rounded(boot.loc[m, "R² high"]),
            "test_median_pct": rounded(scores.loc[m, "median % error"]),
            "test_median_pct_low": rounded(boot.loc[m, "median % low"]),
            "test_median_pct_high": rounded(boot.loc[m, "median % high"]),
            "test_mae_rs": rounded(scores.loc[m, "MAE (₹)"], 0),
            "test_median_abs_error_rs": rounded(scores.loc[m, "median abs error (₹)"], 0),
            "cv_r2": rounded(cv.loc[m, "R2_log_mean"]), "cv_r2_std": rounded(cv.loc[m, "R2_log_std"]),
            "cv_median_pct": rounded(cv.loc[m, "median_pct_error_mean"]),
            "cv_median_pct_std": rounded(cv.loc[m, "median_pct_error_std"]),
            "cv_mae_rs": rounded(cv.loc[m, "MAE_rupees_mean"], 0), "cv_mae_rs_std": rounded(cv.loc[m, "MAE_rupees_std"], 0)}

    within, by_band = {}, {}
    for m in ["Gradient boosting", "Random forest", "Ridge", "Model B: specs + chip + brand"]:
        pct = 100 * (preds[f"₹_pred | {m}"] / actual - 1).abs()
        within[m] = {f"within_{k}": rounded((pct <= k).mean()) for k in [10, 20, 30]}
        bands = actual.map(band_of)
        by_band[m] = {label: {"median_pct": rounded(pct[bands == label].median()),
                              "mae_rs": rounded((preds[f"₹_pred | {m}"] - actual).abs()[bands == label].mean(), 0),
                              "phones": int((bands == label).sum())} for _, _, label in PRICE_BANDS}

    # Robustness (as in notebook 06): rescore the saved test predictions without phones whose India price is doubtful
    flag = test["india_price_flag"].fillna("").reindex(preds.index) != ""
    without_flagged = {}
    for m in ["Ridge", "Gradient boosting"]:
        keep = ~flag
        log_true, log_pred = np.log(actual[keep]), np.log(preds.loc[keep, f"₹_pred | {m}"])
        r2 = 1 - ((log_true - log_pred) ** 2).sum() / ((log_true - log_true.mean()) ** 2).sum()
        without_flagged[m] = {"test_phones": int(keep.sum()), "r2": rounded(r2),
                              "median_pct": rounded(100 * (preds.loc[keep, f"₹_pred | {m}"] / actual[keep] - 1).abs().median())}

    # Error analysis on the test set: median % error by brand (brands with >= 5 test phones) ...
    pct = {m: 100 * (preds[f"₹_pred | {m}"] / actual - 1).abs() for m in ["Ridge", "Gradient boosting"]}
    counts = preds["brand"].value_counts()
    error_by_brand = {
        "min_phones": MIN_BRAND_PHONES,
        "brands": [{"brand": b, "phones": int(n), "ridge_median_pct": rounded(pct["Ridge"][preds["brand"] == b].median()),
                    "boosting_median_pct": rounded(pct["Gradient boosting"][preds["brand"] == b].median())}
                   for b, n in counts.items() if n >= MIN_BRAND_PHONES],
        "too_few": [{"brand": b, "phones": int(n)} for b, n in counts.items() if n < MIN_BRAND_PHONES],
    }
    # ... and the largest misses of the most accurate model (signed: + = predicted too expensive)
    best = scores["median % error"].idxmin()
    signed = {m: 100 * (preds[f"₹_pred | {m}"] / actual - 1) for m in [best, "Ridge"]}
    chips = test["chip_group"].reindex(preds.index)
    largest_misses = [{"phone": f'{preds.loc[i, "brand"]} {preds.loc[i, "Model Name"]}', "brand": preds.loc[i, "brand"],
                       "chip_group": chips[i], "actual": rounded(actual[i], 0),
                       "predicted": rounded(preds.loc[i, f"₹_pred | {best}"], 0),
                       "error_pct": rounded(signed[best][i]), "ridge_error_pct": rounded(signed["Ridge"][i])}
                      for i in signed[best].abs().sort_values(ascending=False).index[:N_LARGEST_MISSES]]

    paired = {}
    for name, r in boot[boot["kind"] == "paired difference"].iterrows():
        paired[name] = {"delta_median_pct": rounded(r["Δ median % (test)"]),
                        "low": rounded(r["Δ median % low"]), "high": rounded(r["Δ median % high"]),
                        "share_first_better": rounded(r["share 1st better (median %)"], 3),
                        "delta_r2": rounded(r["Δ R² (test)"]), "delta_r2_low": rounded(r["Δ R² low"]),
                        "delta_r2_high": rounded(r["Δ R² high"])}

    steps = pd.read_csv(RESULTS / "cleaning_steps.csv")
    all_phones = pd.concat([train, test])
    return {
        "data": {"raw_rows": int(steps["rows_left"].iloc[0]),
                 "phones": len(train) + len(test), "train_phones": len(train), "test_phones": len(test),
                 "train_models": int(train["model_group"].nunique()), "test_models": int(test["model_group"].nunique()),
                 "brands": int(train["brand"].nunique()), "chip_groups": int(train["chip_group"].nunique()),
                 "web_lookup_phones": int((all_phones["storage_source"] == "web_lookup").sum()),
                 "processor_names": int(all_phones["Processor"].nunique()),
                 "flagged_phones": int((all_phones["india_price_flag"].fillna("") != "").sum()),
                 "flagged_test_phones": int((test["india_price_flag"].fillna("") != "").sum()),
                 "year_min": int(all_phones["year"].min()), "year_max": int(all_phones["year"].max()),
                 "price_min": float(all_phones["price_inr"].min()), "price_max": float(all_phones["price_inr"].max()),
                 "price_median": float(all_phones["price_inr"].median())},
        "cleaning": steps.to_dict(orient="records"),
        "method": {"cv_folds": N_FOLDS, "test_share": TEST_SIZE, "bootstrap_resamples": 1000},
        "best_model": scores["median % error"].idxmin(),
        "interpretable_model": "Ridge",
        "models": models,
        "within": within,
        "error_by_band": by_band,
        "paired_differences": paired,
        "without_flagged": without_flagged,
        "error_by_brand": error_by_brand,
        "largest_misses": largest_misses,
    }


# ----------------------------------------------------------------------------- figures
LAYOUT = dict(template="plotly_white", margin=dict(l=60, r=20, t=50, b=50), font=dict(size=13))


def save_figure(fig, name):
    path = FIGURES / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_json(path)
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


def export_figures(all_phones, results, premiums, pca):
    # Phase 2: correlation of each numeric feature with log price (all 702 phones, as in notebook 02)
    corr = pd.read_csv(RESULTS / "correlations_with_log_price.csv", index_col=0)["r (with web lookup, n=702)"].sort_values()
    fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                           marker_color=["#c0504d" if v < 0 else "#2a78d6" for v in corr.values]))
    fig.update_layout(title="Correlation with log price (all phones)", xaxis_title="correlation r", **LAYOUT)
    save_figure(fig, "explore_correlations")

    # Phase 2: price distribution (bins equally wide on a log scale)
    edges = np.logspace(np.log10(all_phones["price_inr"].min()), np.log10(all_phones["price_inr"].max()), 31)
    counts, _ = np.histogram(all_phones["price_inr"], bins=edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    fig = go.Figure(go.Bar(x=centres, y=counts, width=np.diff(edges), marker_color="#2a78d6",
                           hovertemplate="around ₹%{x:,.0f}: %{y} phones<extra></extra>"))
    fig.update_layout(title="Launch prices (log scale)", xaxis=dict(type="log", title="launch price (₹)"),
                      yaxis_title="number of phones", **LAYOUT)
    save_figure(fig, "explore_price_hist")

    # Phase 2: price by brand and by chip group (box plots, log scale, ordered by median)
    for col, title, name in [("brand", "Launch price by brand", "explore_price_by_brand"),
                             ("chip_group", "Launch price by chip group", "explore_price_by_chip")]:
        order = all_phones.groupby(col)["price_inr"].median().sort_values().index
        fig = go.Figure()
        for value in order:
            prices = all_phones.loc[all_phones[col] == value, "price_inr"]
            fig.add_trace(go.Box(y=prices, name=f"{value} ({len(prices)})", marker_color="#2a78d6", showlegend=False))
        fig.update_layout(title=title, yaxis=dict(type="log", title="launch price (₹)"), **LAYOUT)
        save_figure(fig, name)

    # PCA scree plot: share of the variance per component and the running total
    comps = pca["components"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[c["name"] for c in comps], y=[100 * c["share"] for c in comps], name="share of variance",
                         marker_color="#2a78d6", hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[c["name"] for c in comps], y=[100 * c["cumulative"] for c in comps], name="cumulative share",
                             mode="lines+markers", marker_color="#eb6834", hovertemplate="up to %{x}: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="PCA of the 9 specs: variance per component", yaxis=dict(title="% of total variance", range=[0, 105]),
                      **LAYOUT)
    save_figure(fig, "pca_variance")

    # Model comparison: test median % error with 95% interval, CV value as a second marker
    ms = sorted(results["models"], key=lambda m: results["models"][m]["test_median_pct"], reverse=True)
    t = [results["models"][m] for m in ms]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["test_median_pct"] for r in t], y=ms, mode="markers", name="test set (95% interval)",
        marker=dict(size=10, color="#2a78d6"),
        error_x=dict(type="data", symmetric=False, array=[r["test_median_pct_high"] - r["test_median_pct"] for r in t],
                     arrayminus=[r["test_median_pct"] - r["test_median_pct_low"] for r in t])))
    fig.add_trace(go.Scatter(x=[r["cv_median_pct"] for r in t], y=ms, mode="markers", name="cross-validation",
                             marker=dict(size=9, symbol="diamond-open", color="#eb6834")))
    fig.update_layout(title="Median % price error (lower is better)", xaxis_title="median % error",
                      height=480, **LAYOUT)
    save_figure(fig, "model_comparison")

    # Predicted vs actual (Phase 7 figure, copied unchanged)
    shutil.copy(RESULTS / "fig07_pred_vs_actual.json", FIGURES / "pred_vs_actual.json")
    print("wrote docs/figures/pred_vs_actual.json  (copy of results/fig07_pred_vs_actual.json)")

    # Error by price band, ridge vs gradient boosting
    fig = go.Figure()
    for m, colour in [("Ridge", "#9aa0a6"), ("Gradient boosting", "#2a78d6")]:
        bands = results["error_by_band"][m]
        fig.add_trace(go.Bar(x=list(bands), y=[b["median_pct"] for b in bands.values()], name=m, marker_color=colour,
                             customdata=[b["phones"] for b in bands.values()],
                             hovertemplate="%{x}: %{y:.1f}% (%{customdata} test phones)<extra>" + m + "</extra>"))
    fig.update_layout(title="Median % error by price band (test set)", yaxis_title="median % error",
                      barmode="group", **LAYOUT)
    save_figure(fig, "error_by_band")

    # Brand premiums with intervals
    p = sorted(premiums, key=lambda r: r["premium_pct"])
    fig = go.Figure(go.Bar(
        x=[r["premium_pct"] for r in p], y=[f"{r['label']} (n={r['n_train']})" for r in p], orientation="h",
        marker_color=["#b9bcc0" if r["includes_zero"] else "#2a78d6" for r in p],
        error_x=dict(type="data", symmetric=False, array=[r["high_pct"] - r["premium_pct"] for r in p],
                     arrayminus=[r["premium_pct"] - r["low_pct"] for r in p]),
        hovertemplate="%{y}: %{x:+.0f}%<extra></extra>"))
    fig.update_layout(title="Brand premium vs the average brand, same specs and chip group",
                      xaxis_title="% price difference vs an average brand (grey bars: interval includes 0%)", height=520, **LAYOUT)
    save_figure(fig, "brand_premiums")


def main():
    train, test = load_train_test(include_web_lookup=True)
    all_phones = pd.concat([train, test])
    write_json(export_models(train), DOCS / "model.json")
    boosting = DOCS / "boosting.json"   # written compactly: 200 trees
    boosting.write_text(json.dumps(export_boosting(), separators=(",", ":")), encoding="utf-8")
    print(f"wrote {boosting.relative_to(ROOT)}  ({boosting.stat().st_size / 1024:.0f} KB)")
    write_json(export_phones(), DOCS / "phones.json")
    premiums = export_premiums()
    write_json(premiums, DOCS / "brand_premiums.json")
    write_json(export_spec_values(), DOCS / "spec_values.json")
    write_json(export_ols(), DOCS / "ols.json")
    pca = export_pca()
    write_json(pca, DOCS / "pca.json")
    results = export_results(train, test)
    write_json(results, DOCS / "results.json")
    export_figures(all_phones, results, premiums, pca)


if __name__ == "__main__":
    main()
