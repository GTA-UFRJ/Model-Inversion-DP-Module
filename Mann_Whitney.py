import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import argparse


# --- CONFIG -------------------------------------------------------------
CSV_PATH = "new_results_with_seed.csv"     # ← mude o nome do arquivo se for diferente
# ------------------------------------------------------------------------

# --- Processamento de Argumentos ---
parser = argparse.ArgumentParser(
    description='''Realiza o teste de Mann-Whitney U para comparar a significância estatística 
                   entre a acurácia do modelo sem ruído e com um nível de ruído selecionado.''',
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('--selected_noise', type=float, required=True,
                    help='''O nível de ruído a ser comparado com o nível de ruído 0.0.
O teste verifica se há uma diferença estatisticamente significativa 
na acurácia do modelo entre esses dois cenários.''')
args = parser.parse_args()
SELECTED_NOISE = args.selected_noise

# 1) Carregamento do CSV
df = pd.read_csv(CSV_PATH)

# 2) Filtragem de linhas
# Fixando a Label, o Round e o Método para consistência 
mask = (df["Label"] == 0) & \
       (df["Round"] == 50) & \
       (df["Method"] == "loss") & \
       (df["Noise Scale"] < 0.11)
subset = df.loc[mask, ["Noise Scale", "Accuracy"]].copy()

# 3) Conversão para numérico
subset["Noise Scale"] = pd.to_numeric(subset["Noise Scale"], errors="coerce")
subset["Accuracy"] = pd.to_numeric(subset["Accuracy"], errors="coerce")

# 4) Normalizando acurácia para [0, 1]
subset["Accuracy"] /= 100

# Assumindo que você tem o DataFrame "subset"
group1 = subset[subset["Noise Scale"] == 0.0]["Accuracy"]
group2 = subset[subset["Noise Scale"] == SELECTED_NOISE]["Accuracy"]

if len(group2) == 0:
    print(f"Erro: Nenhum dado encontrado para o nível de ruído {SELECTED_NOISE}.")
    print("Níveis de ruído disponíveis no dataset filtrado:")
    print(sorted(subset["Noise Scale"].unique()))
    exit()

# Teste de Mann-Whitney
stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

print(f"Comparando Ruído 0.0 com Ruído {SELECTED_NOISE}")
print(f"Mann-Whitney U statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
print("É significante a uma incerteza de 0.05?" , p_value < 0.05)
