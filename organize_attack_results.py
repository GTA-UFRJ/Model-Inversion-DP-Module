import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG -------------------------------------------------------------
CSV_PATH = "new_results_with_seed.csv"     # ← change to your file name if different
OUT_PNG  = "graficos/Acurácia por Ruido.png"
# ------------------------------------------------------------------------

# 1) Load the CSV
df = pd.read_csv(CSV_PATH)

# Helper function to plot and save MSE for Noise Scale 50
def plot_MSE(data, method, title, filename):

    filtered = data[(data['Round'] == 50) & (data['Method'] == method) & (data['Noise Scale'] == 0.0)]
    
    # Group by Label and take mean MSE (in case there are multiple entries)

    grouped = filtered.groupby('Label')['MSE'].mean()

    
    plt.figure()
    grouped.plot(kind='bar', rot=0)
    plt.xlabel('Classe', fontsize=25)
    plt.ylabel('MSE', fontsize=25)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Plot and save for each requested condition
plot_MSE(df, method='loss', title='MSE for Min Client - Loss Attack at Round 50', filename='graficos/MSE_por_classe.png')


def plot_MSE_over_Noise_Scales(data, filename):

    filtered = data[(data['Round'] == 50) & (data['Noise Scale'] < 0.06)]

    # Mapeamento de nomes e estilos
    name_mapping = {'loss': 'Gradiente', 'naive': 'Ingênua'}
    style_mapping = {'loss': '-', 'naive': '--'}

    # Agrupa por Noise Scale e Method, tirando a média do MSE (média sobre todas as labels)

    grouped = filtered.groupby(['Noise Scale', 'Method'])['MSE'].mean().reset_index()


    plt.figure()
    for method in grouped['Method'].unique():
        method_data = grouped[grouped['Method'] == method]
        plt.plot(
            method_data['Noise Scale'], 
            method_data['MSE'], 

            label=name_mapping.get(method, method), 
            linestyle=style_mapping.get(method, '-'),
            linewidth=2.5  # Aumenta a espessura da linha
        )
    
    plt.xlabel('Ruído', fontsize=25)
    plt.ylabel('MSE', fontsize=25)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

plot_MSE_over_Noise_Scales(df, filename='graficos/MSE_por_ruido_médio.png')


# Helper function to plot and save PSNR for Noise Scale 50
def plot_PSNR(data, method, title, filename):

    filtered = data[(data['Round'] == 50) & (data['Method'] == method) & (data['Noise Scale'] == 0.0)]
    
    # Group by Label and take mean PSNR (in case there are multiple entries)

    grouped = filtered.groupby('Label')['PSNR'].mean()

    
    plt.figure()
    grouped.plot(kind='bar', rot=0)
    plt.xlabel('Classe', fontsize=25)
    plt.ylabel('PSNR', fontsize=25)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Plot and save for each requested condition
plot_PSNR(df, method='loss', title='PSNR for Min Client - Loss Attack at Round 50', filename='graficos/PSNR_por_classe.png')


def plot_PSNR_over_Noise_Scales(data, filename):

    filtered = data[(data['Round'] == 50) & (data['Noise Scale'] < 0.06)]

    # Mapeamento de nomes e estilos
    name_mapping = {'loss': 'Gradiente', 'naive': 'Ingênua'}
    style_mapping = {'loss': '-', 'naive': '--'}

    # Agrupa por Noise Scale e Method, tirando a média do PSNR (média sobre todas as labels)

    grouped = filtered.groupby(['Noise Scale', 'Method'])['PSNR'].mean().reset_index()


    plt.figure()
    for method in grouped['Method'].unique():
        method_data = grouped[grouped['Method'] == method]
        plt.plot(
            method_data['Noise Scale'], 
            method_data['PSNR'], 

            label=name_mapping.get(method, method), 
            linestyle=style_mapping.get(method, '-'),
            linewidth=2.5  # Aumenta a espessura da linha
        )
    
    plt.xlabel('Ruído', fontsize=25)
    plt.ylabel('PSNR', fontsize=25)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

plot_PSNR_over_Noise_Scales(df, filename='graficos/PSNR_por_ruido_médio.png')

# Helper function to plot and save SSIM for Noise Scale 50
def plot_SSIM(data, method, title, filename):

    filtered = data[(data['Round'] == 50) & (data['Method'] == method) & (data['Noise Scale'] == 0.0)]
    
    # Group by Label and take mean SSIM (in case there are multiple entries)

    grouped = filtered.groupby('Label')['SSIM'].mean()

    
    plt.figure()
    grouped.plot(kind='bar', rot=0)
    plt.xlabel('Classe', fontsize=25)
    plt.ylabel('SSIM', fontsize=25)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Plot and save for each requested condition
plot_SSIM(df, method='loss', title='SSIM for Min Client - Loss Attack at Round 50', filename='graficos/SSIM_por_classe.png')


def plot_SSIM_over_Noise_Scales(data, filename):

    filtered = data[(data['Round'] == 50) & (data['Noise Scale'] < 0.06)]

    # Mapeamento de nomes e estilos
    name_mapping = {'loss': 'Gradiente', 'naive': 'Ingênua'}
    style_mapping = {'loss': '-', 'naive': '--'}

    # Agrupa por Noise Scale e Method, tirando a média do SSIM (média sobre todas as labels)

    grouped = filtered.groupby(['Noise Scale', 'Method'])['SSIM'].mean().reset_index()


    plt.figure()
    for method in grouped['Method'].unique():
        method_data = grouped[grouped['Method'] == method]
        plt.plot(
            method_data['Noise Scale'], 
            method_data['SSIM'], 

            label=name_mapping.get(method, method), 
            linestyle=style_mapping.get(method, '-'),
            linewidth=2.5  # Aumenta a espessura da linha
        )
    
    plt.xlabel('Ruído', fontsize=25)
    plt.ylabel('SSIM', fontsize=25)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

plot_SSIM_over_Noise_Scales(df, filename='graficos/SSIM_por_ruido_médio.png')
