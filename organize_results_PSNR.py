import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV data
df = pd.read_csv('new_results_with_seed.csv')  # Replace with your actual filename

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
