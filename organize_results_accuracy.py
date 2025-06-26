import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG -------------------------------------------------------------
CSV_PATH = "new_results_with_seed.csv"     # ← change to your file name if different
OUT_PNG  = "Acurácia por Ruido.png"
# ------------------------------------------------------------------------

# 1) Load the CSV
df = pd.read_csv(CSV_PATH)

# 2) Filter relevant rows
mask = (df["Label"] == 0) & \
       (df["Round"] == 50) & \
       (df["Method"] == "loss") & \
       (df["Noise Scale"] < 0.11)
subset = df.loc[mask, ["Noise Scale", "Accuracy"]].copy()

# 3) Convert to numeric
subset["Noise Scale"] = pd.to_numeric(subset["Noise Scale"], errors="coerce")
subset["Accuracy"] = pd.to_numeric(subset["Accuracy"], errors="coerce")

# 4) Normalize accuracy to [0, 1]
subset["Accuracy"] /= 100

# 5) Group by Noise Scale
grouped = subset.groupby("Noise Scale")["Accuracy"]
means = grouped.mean()
stds = grouped.std()
counts = grouped.count()
print(counts)
# Corrigir os desvios padrão para erro padrão da média (±1.96 * std / sqrt(n))
stds = 1.96 * stds / counts.pow(0.5)


# 6) Plot with error bars
plt.figure(figsize=(8, 5))
plt.errorbar(
    means.index, means.values, yerr=stds.values,
    fmt='o-', capsize=5
)

# 7) Labels and formatting
plt.xlabel("Ruído", fontsize=16)
plt.ylabel("Acurácia média", fontsize=16)
plt.ylim(0.5, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()
