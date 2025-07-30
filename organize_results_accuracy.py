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

from scipy.stats import f_oneway
import pandas as pd

# Supondo que você já tenha o DataFrame original "subset" com colunas:
# "Noise Scale" e "Accuracy"

# Ordenar os níveis de ruído
noise_levels = sorted(subset["Noise Scale"].unique())

# Aplicar ANOVA entre pares adjacentes
anova_results = []

for i in range(len(noise_levels) - 1):
    group1 = subset[subset["Noise Scale"] == noise_levels[i]]["Accuracy"]
    group2 = subset[subset["Noise Scale"] == noise_levels[i+1]]["Accuracy"]
    
    f_stat, p_value = f_oneway(group1, group2)
    
    anova_results.append({
        "Noise Scale 1": noise_levels[i],
        "Noise Scale 2": noise_levels[i+1],
        "F-statistic": f_stat,
        "p-value": p_value,
        "Significant at 0.05": p_value < 0.05
    })

anova_df = pd.DataFrame(anova_results)
anova_df