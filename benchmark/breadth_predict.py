"""
Breadth benchmark — a-priori predictions and dataset registry.

Each dataset has a prediction made BEFORE running:
  REGEN  — heterogeneous features → ARD+Scout advantage
  SMOTE  — homogeneous/compressed features → SMOTE wins

Commit prediction before running; test the hypothesis.
"""
import json

DATASETS = [
    # ── Existing (3) ──────────────────────────────────────────────────
    {
        "id": 40676, "name": "hypothyroid",
        "display": "Hypothyroid",
        "label": "Class", "rare_value": 0,
        "rows": 3163, "minority": 151, "minority_pct": 4.77,
        "features": 25, "regime": "heterogeneous",
        "composition": "mixed continuous + 17 binary flags",
        "prediction": "REGEN",
        "note": "Existing win from multi-pass benchmark (+10%)",
    },
    {
        "id": 40900, "name": "satellite",
        "display": "Satellite",
        "label": "Target", "rare_value": 1,
        "rows": 5100, "minority": 75, "minority_pct": 1.47,
        "features": 36, "regime": "homogeneous",
        "composition": "all-continuous remote-sensing bands",
        "prediction": "SMOTE",
        "note": "Existing REGEN win (+26%) — tests if homogeneity rule holds or fails",
    },
    {
        "id": 42175, "name": "creditcard",
        "display": "Credit Card Fraud",
        "label": "Class", "rare_value": 1,
        "rows": 284807, "minority": 492, "minority_pct": 0.17,
        "features": 30, "regime": "homogeneous",
        "composition": "28 PCA components + Amount + Time",
        "prediction": "SMOTE",
        "note": "Existing SMOTE win — baseline loss case",
    },
    # ── New heterogeneous (predicted REGEN-favorable) ────────────────
    {
        "id": 4135, "name": "amazon",
        "display": "Amazon Employee Access",
        "label": "ACTION", "rare_value": 1,
        "rows": 32769, "minority": 1897, "minority_pct": 5.79,
        "features": 9, "regime": "heterogeneous",
        "composition": "mixed categorical + numeric (HR access prediction)",
        "prediction": "REGEN",
        "note": "9 features with varying informativeness — ideal for ARD targeting",
    },
    {
        "id": 40983, "name": "wilt",
        "display": "Wilt (Remote Sensing)",
        "label": "class", "rare_value": "w",
        "rows": 4839, "minority": 261, "minority_pct": 5.39,
        "features": 5, "regime": "heterogeneous",
        "composition": "5 mixed features (satellite + soil data)",
        "prediction": "REGEN",
        "note": "Few features — ARD has less room but still heterogeneous",
    },
    {
        "id": 1461, "name": "bank_marketing",
        "display": "Bank Marketing",
        "label": "y", "rare_value": "yes",
        "rows": 45211, "minority": 5289, "minority_pct": 11.70,
        "features": 16, "regime": "heterogeneous",
        "composition": "7 numeric + 9 categorical (demographics + campaign data)",
        "prediction": "REGEN",
        "note": "Classic UCI benchmark, mixed types, varying signal",
    },
    {
        "id": 42738, "name": "open_payments",
        "display": "Open Payments",
        "label": "Class", "rare_value": 1,
        "rows": 73558, "minority": 4749, "minority_pct": 6.46,
        "features": 5, "regime": "heterogeneous",
        "composition": "5 mixed features (medical payment data)",
        "prediction": "REGEN",
        "note": "Few features but mixed types — ARD can still differentiate",
    },
    {
        "id": 40701, "name": "churn",
        "display": "Churn (Telecom)",
        "label": "Class", "rare_value": 1,
        "rows": 5000, "minority": 707, "minority_pct": 14.14,
        "features": 20, "regime": "heterogeneous",
        "composition": "mixed numeric + categorical (customer churn)",
        "prediction": "REGEN",
        "note": "Customer churn — informative features vary widely",
    },
    # ── New homogeneous (predicted SMOTE-favorable) ─────────────────
    {
        "id": 4154, "name": "creditcard_subset",
        "display": "CreditCard Subset",
        "label": "Class", "rare_value": 1,
        "rows": 14240, "minority": 23, "minority_pct": 0.16,
        "features": 30, "regime": "homogeneous",
        "composition": "28 PCA components + Amount + Time (subset of main)",
        "prediction": "SMOTE",
        "note": "Same PCA structure as creditcard — tests if result generalizes",
    },
    {
        "id": 1487, "name": "ozone",
        "display": "Ozone Level (8hr)",
        "label": "Class", "rare_value": 1,
        "rows": 2534, "minority": 160, "minority_pct": 6.31,
        "features": 72, "regime": "homogeneous",
        "composition": "72 all-continuous atmospheric measurements",
        "prediction": "SMOTE",
        "note": "High-dim all-continuous — strong test of homogeneity hypothesis",
    },
    {
        "id": 40702, "name": "solar_flare",
        "display": "Solar Flare",
        "label": "Class", "rare_value": 1,
        "rows": 1066, "minority": 182, "minority_pct": 17.07,
        "features": 10, "regime": "heterogeneous",
        "composition": "10 mixed features (solar physics)",
        "prediction": "REGEN",
        "note": "Small dataset but heterogeneous features",
    },
]

print("A-PRIORI PREDICTIONS (before running any benchmark)")
print("=" * 60)
print(f"{'Dataset':30s} {'Regime':15s} {'Prediction':10s}")
print("-" * 60)
for d in DATASETS:
    sym = "🔮" if d["prediction"] == "REGEN" else "⚡"
    print(f"{d['display']:30s} {d['regime']:15s} {sym} {d['prediction']:8s}")

regen_pred = sum(1 for d in DATASETS if d["prediction"] == "REGEN")
smote_pred = sum(1 for d in DATASETS if d["prediction"] == "SMOTE")
print(f"\nPredictions: {regen_pred} REGEN, {smote_pred} SMOTE")

with open("benchmark/breadth_predictions.json", "w") as f:
    json.dump(DATASETS, f, indent=2)
print("Saved to benchmark/breadth_predictions.json")
