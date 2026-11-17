"""Check that docs/model.json and docs/boosting.json reproduce the Python models (Phase 9).

The website computes predictions in JavaScript from these files. This test does
the same arithmetic in plain Python (math and struct only, no NumPy) and compares
it with the fitted models saved in Phase 7 (sklearn pipelines in models/):
ridge and Model A for 20 phones, gradient boosting for 20 phones and for all phones.

Run:  python tests/test_export.py      (or: pytest tests/test_export.py)
"""
import json
import math
import struct
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import NUMERIC_FEATURES  # noqa: E402

TOLERANCE = 1e-6   # on log price


def load():
    model = json.load(open(ROOT / "docs" / "model.json", encoding="utf-8"))
    boosting = json.load(open(ROOT / "docs" / "boosting.json", encoding="utf-8"))
    phones = json.load(open(ROOT / "docs" / "phones.json", encoding="utf-8"))
    sample = phones[::len(phones) // 20][:20]   # 20 phones spread over the file (train and test)
    return model, boosting, phones, sample


def features_from_inputs(model, phone):
    """Raw inputs (RAM in GB, ...) -> model features (log2_ram, ...), as the website does."""
    x = {}
    for feature, t in model["transforms"].items():
        value = phone[t["from"]]
        x[feature] = math.log2(value) if t["transform"] == "log2" else value
    return x


def ridge_by_hand(model, phone):
    r = model["ridge"]
    x = features_from_inputs(model, phone)
    total = r["intercept"]
    for name, mean, scale, coef in zip(r["numeric_features"], r["means"], r["scales"], r["numeric_coefs"]):
        total += coef * (x[name] - mean) / scale
    for col in ["chip_group", "brand"]:
        total += r["categories"][col].get(phone[col], 0.0)   # unseen category -> 0
    return total


def model_a_by_hand(model, phone):
    a = model["model_a"]
    x = features_from_inputs(model, phone)
    total = a["intercept"]
    for name, coef in zip(a["numeric_features"], a["numeric_coefs"]):
        total += coef * x[name]
    return total + a["categories"]["chip_group"].get(phone["chip_group"], 0.0)


def float32(v):
    """Round to the nearest 32-bit float, as sklearn's trees do with their inputs (JavaScript: Math.fround)."""
    return struct.unpack("f", struct.pack("f", v))[0]


def boosting_by_hand(model, boosting, phone):
    x = features_from_inputs(model, phone)
    row = [x[name] for name in boosting["numeric_features"]]
    for col in ["chip_group", "brand"]:
        row += [1.0 if phone[col] == c else 0.0 for c in boosting["categories"][col]]   # unseen -> all zeros
    total = boosting["init"]
    for tree in boosting["trees"]:
        node = 0
        while tree["left"][node] != -1:   # walk down to a leaf
            f = tree["feature"][node]
            node = tree["left"][node] if float32(row[f]) <= tree["threshold"][node] else tree["right"][node]
        total += boosting["learning_rate"] * tree["value"][node]
    return total


def python_predictions(phones):
    """The same phones through the saved sklearn pipelines from Phase 7."""
    rows = pd.DataFrame(phones)
    rows["log2_ram"], rows["log2_storage"] = rows["ram_gb"].map(math.log2), rows["storage_gb"].map(math.log2)
    X = rows[NUMERIC_FEATURES + ["chip_group", "brand"]]
    ridge = joblib.load(ROOT / "models" / "ridge.joblib")
    model_a = joblib.load(ROOT / "models" / "model_a_ols.joblib")
    boosting = joblib.load(ROOT / "models" / "gradient_boosting.joblib")
    return ridge.predict(X), model_a.predict(X), boosting.predict(X)


def compare():
    model, boosting, _, phones = load()
    ridge_py, a_py, gb_py = python_predictions(phones)
    rows = []
    for phone, rp, ap, gp in zip(phones, ridge_py, a_py, gb_py):
        rh, ah, gh = ridge_by_hand(model, phone), model_a_by_hand(model, phone), boosting_by_hand(model, boosting, phone)
        rows.append({"phone": phone["name"], "set": phone["set"], "ridge (JSON)": rh, "ridge (sklearn)": rp,
                     "Model A (JSON)": ah, "Model A (sklearn)": ap, "boosting (JSON)": gh, "boosting (sklearn)": gp,
                     "max abs diff": max(abs(rh - rp), abs(ah - ap), abs(gh - gp))})
    return pd.DataFrame(rows)


def boosting_all_phones():
    """Largest difference between the hand-walked trees and sklearn over every phone."""
    model, boosting, phones, _ = load()
    gb_py = python_predictions(phones)[2]
    return max(abs(boosting_by_hand(model, boosting, p) - g) for p, g in zip(phones, gb_py))


def test_json_matches_python_models():
    table = compare()
    assert len(table) == 20
    assert table["max abs diff"].max() < TOLERANCE, table.sort_values("max abs diff").tail()


def test_boosting_matches_on_all_phones():
    assert boosting_all_phones() < TOLERANCE


if __name__ == "__main__":
    table = compare()
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    worst = max(table["max abs diff"].max(), boosting_all_phones())
    print(f"\nlargest difference (20 phones, all three models; boosting also on all phones): {worst:.2e} "
          f"(tolerance {TOLERANCE:g}) -> {'PASS' if worst < TOLERANCE else 'FAIL'}")
    sys.exit(0 if worst < TOLERANCE else 1)
