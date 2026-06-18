"""
Find imbalanced binary datasets — skip task listing, check datasets directly.
"""
import openml, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

candidates = [
    1461, 1464, 1466, 1468, 1471, 1475, 1476, 1478, 1479,
    1480, 1485, 1486, 1487, 1489, 1491, 1494, 1497, 1501,
    1510, 1515, 1590, 4134, 4135, 4136, 4137, 4142, 4144,
    4146, 4148, 4150, 4154, 4155, 4156, 4157, 4158, 42125,
    42730, 42731, 42732, 42733, 42734, 42736, 42738, 42742,
    42744, 42803, 40966, 40967, 40968, 40969, 40970, 40983,
    40984, 40675, 40677, 40678, 40680, 40681, 40682, 40685,
    40687, 40688, 40690, 40692, 40693, 40694, 40696, 40697,
    40698, 40699, 40700, 40701, 40702, 40703, 42175, 40676, 40900,
]

found = []
for oid in sorted(set(candidates)):
    try:
        d = openml.datasets.get_dataset(oid, download_data=True, download_qualities=False)
        X, y, _, _ = d.get_data(target=d.default_target_attribute)
        if y is None:
            continue
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = pd.Series(y).dropna()
        if len(y) < 200:
            continue
        counts = y.value_counts()
        if len(counts) != 2:
            continue
        minority_pct = counts.min() / len(y) * 100
        n_min = counts.min()
        if minority_pct > 20 or n_min < 20:
            continue
        # Feature type analysis
        if isinstance(X, pd.DataFrame):
            n_cols = len(X.columns)
            n_num = sum(1 for c in X.columns if pd.api.types.is_numeric_dtype(X[c]))
            n_cat = n_cols - n_num
        else:
            n_cols = X.shape[1]
            n_num = n_cols
            n_cat = 0
        # Estimate feature informativeness heterogeneity
        # If mixed numeric+categorical or all-numeric with varying distributions → heterogeneous
        # If PCA'd or all-continuous-similar-scale → homogeneous
        if n_cat > 0:
            regime = "heterogeneous"
        elif n_cols >= 10 and n_num == n_cols:
            # All numeric — check if likely PCA/embeddings (continuous features named V1,V2)
            if isinstance(X, pd.DataFrame):
                col_names = list(X.columns)
                pca_pattern = sum(1 for c in col_names if c.startswith("V") and c[1:].isdigit())
                if pca_pattern > n_cols * 0.5:
                    regime = "homogeneous"
                else:
                    regime = "heterogeneous"
            else:
                regime = "heterogeneous"
        else:
            regime = "heterogeneous"
        
        found.append({
            "id": oid, "name": d.name, "rows": len(y), "minority": n_min,
            "minority_pct": round(minority_pct, 2),
            "features": n_cols, "numeric": n_num, "categorical": n_cat,
            "regime": regime,
        })
        print(f"  {oid:>6} {d.name:30s} {regime:15s} {len(y):>7} rows, {n_min:>4} minority ({minority_pct:.2f}%)", flush=True)
    except Exception as e:
        pass

# Save
df = pd.DataFrame(found).sort_values("minority_pct")
df.to_csv("benchmark/dataset_candidates.csv", index=False)
print(f"\n{'='*60}")
print(f"Found {len(df)} suitable datasets")
print(f"  Heterogeneous: {len(df[df['regime']=='heterogeneous'])}")
print(f"  Homogeneous:   {len(df[df['regime']=='homogeneous'])}")
print(f"Saved to benchmark/dataset_candidates.csv")

# Show table
print(f"\n{'ID':>7} {'Name':30s} {'Regime':15s} {'Rows':>7} {'Min':>5} {'%':>6} {'Feat':>5}")
print("-" * 80)
for _, r in df.iterrows():
    print(f"{r['id']:>7} {r['name']:30s} {r['regime']:15s} {r['rows']:>7} {r['minority']:>5} {r['minority_pct']:>5.2f}% {r['features']:>5}")
