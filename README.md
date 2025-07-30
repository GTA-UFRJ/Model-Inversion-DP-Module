# Título projeto

**Título do Artigo:** Avaliação da Privacidade Diferencial Aplicada ao Aprendizado Federado Através de Ataques de Inversão de Modelo

**Resumo:** Este trabalho propõe uma metodologia prática para avaliar a eficácia de mecanismos de Privacidade Diferencial (DP) no Aprendizado Federado (FL) diante de ataques de inversão de modelo. Para isso, foi adotado um cenário adversarial baseado no paradigma de segurança cibernética equipe vermelha/equipe azul (RT/BT), onde a equipe azul implementa a proteção de DP e a equipe vermelha realiza ataques para recuperar dados de um cliente específico a partir do modelo global. Foram implementados dois tipos de ataques — um baseado em gradientes e outro ingênuo — aplicados em diferentes intensidades de ruído gaussiano. Os experimentos demonstraram que ruídos baixos já são suficientes para mitigar significativamente os ataques de inversão, conforme evidenciado tanto por métricas quantitativas (SSIM, PSNR e MSE) quanto pela análise visual das imagens reconstruídas. Por outro lado, observou-se que a acurácia do modelo global se mantém estável até níveis moderados de ruído, com impacto limitado no desempenho. Esses resultados indicam que é possível alcançar um bom equilíbrio entre privacidade e utilidade do modelo em cenários práticos.

# Estrutura do readme.md

Este `README.md` está organizado nas seguintes seções para guiar os avaliadores na replicação dos resultados:

*   **Título projeto:** Apresenta o título e o resumo do artigo.
*   **Estrutura do readme.md:** Descreve a organização deste documento.
*   **Selos Considerados:** Lista os selos de qualidade almejados.
*   **Informações básicas:** Detalha os requisitos de hardware e software.
*   **Dependências:** Lista as bibliotecas e outras dependências necessárias.
*   **Obtenção do Conjunto de Dados:** Instruções para baixar o dataset MNIST.
*   **Preocupações com segurança:** Informa sobre possíveis riscos.
*   **Instalação:** Guia passo a passo para configurar o ambiente.
*   **Teste mínimo:** Um teste rápido para verificar a instalação.
*   **Experimentos:** Instruções detalhadas para reproduzir as principais reivindicações do artigo.
*   **LICENSE:** A licença do software.

# Selos Considerados

Os selos considerados para este artefato são: **Disponível (SELOD)**, **Funcional (SELOF)**, **Sustentável (SELOS)** e **Reprodutível (SELOR)**.

# Informações básicas

*   **Hardware:** Um computador padrão com pelo menos 8 GB de RAM e 200 MB de espaço em disco.
*   **Software:**
    *   Sistema Operacional: Windows, macOS ou Linux.
    *   Python 3.8 ou superior.

# Dependências

As dependências de software podem ser instaladas via `pip`. As versões exatas usadas no desenvolvimento estão listadas abaixo, mas versões mais recentes provavelmente funcionarão.

*   `numpy==1.26.4`
*   `pandas==2.2.2`
*   `matplotlib==3.8.4`
*   `scikit-image==0.23.2`
*   `torch==2.3.0`
*   `opencv-python==4.9.0.80`
*   `scipy==1.13.0`

## Obtenção do Conjunto de Dados

O conjunto de dados MNIST, no formato CSV, não está incluído neste repositório devido ao seu tamanho. Você pode baixá-lo do Kaggle no seguinte link:

*   [MNIST in CSV no Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

Faça o download do arquivo `mnist_train.csv` e coloque-o na raiz deste diretório.

# Preocupações com segurança

A execução deste artefato não apresenta riscos de segurança conhecidos. Os scripts apenas realizam cálculos numéricos, manipulam arquivos dentro do diretório do projeto e não acessam informações sensíveis do sistema.

# Instalação

1.  Clone ou baixe este repositório.
2.  Instale as dependências Python listadas na seção **Dependências** usando o `pip`:
    ```bash
    pip install numpy pandas matplotlib scikit-image torch opencv-python scipy
    ```
3.  Baixe o arquivo `mnist_train.csv` conforme as instruções na seção **Obtenção do Conjunto de Dados** e coloque-o na raiz do projeto.

# Teste mínimo

Para garantir que o ambiente está configurado corretamente, você pode executar um teste mínimo. Este teste executa o experimento para apenas uma classe, um nível de ruído e por 10 rodadas.

1.  Abra o script `MNIST_FL_inversion_comparison.py` em um editor de texto.
2.  Localize o laço de repetição principal no final do arquivo.
3.  Altere os parâmetros do `range` para que fiquem como no exemplo abaixo:
    ```python
    if __name__ == "__main__":
        for seed in range(1): # Executa para 1 semente
            for NOISE_SCALEtime100 in range(1): # Executa para 1 nível de ruído
                for i in range(1): # Executa para 1 classe
                    # ... (o resto do código permanece o mesmo)
                    num_of_training_rounds = 10 # Executa por 10 rodadas
    ```
4.  Execute o script:
    ```bash
    python MNIST_FL_inversion_comparison.py
    ```
O teste deve demorar cerca de 1-2 minutos. Se o script for executado sem erros e gerar um arquivo `new_results_with_seed.csv` e algumas imagens na pasta `0/0.0 noise/`, a instalação foi bem-sucedida. **Lembre-se de reverter as alterações no script antes de executar os experimentos completos.**

# Experimentos

Esta seção descreve como reproduzir as principais reivindicações do artigo.

**Passo 0: Execução Completa**

Primeiro, execute o script principal sem modificações para gerar todos os dados.

*   **Comando:** `python MNIST_FL_inversion_comparison.py`
*   **Tempo esperado:** 2-3 horas, dependendo do hardware.
*   **Recursos esperados:** ~1 GB de RAM, ~100 MB de espaço em disco para os resultados.
*   **Resultado esperado:** Um arquivo `new_results_with_seed.csv` e uma estrutura de diretórios com as imagens reconstruídas para cada classe, nível de ruído e rodada.

**Passo 1 (Opcional): Limpeza dos Dados**

Opcionalmente, execute o script `clean bad rows.py` para remover sementes que levaram a resultados anômalos.

*   **Comando:** `python "clean bad rows.py"`
*   **Resultado esperado:** Um arquivo `updated_new_results_with_seed.csv`. Se você executar este passo, renomeie o arquivo de saída para `new_results_with_seed.csv` para os próximos passos.

## Reivindicação #1: Ruídos baixos mitigam significativamente os ataques de inversão.

Esta reivindicação é validada pela análise do erro (MSE) das imagens reconstruídas em função do ruído aplicado.

*   **Comando:** `python organize_results.py`
*   **Tempo esperado:** Menos de 1 minuto.
*   **Resultado esperado:** A geração dos gráficos `MSE_por_classe.png` e `MSE_por_ruido_médio.png`. O gráfico `MSE_por_ruido_médio.png` demonstra visualmente que o MSE aumenta rapidamente com o aumento do ruído, indicando uma queda na qualidade das imagens reconstruídas e, portanto, a mitigação do ataque.

## Reivindicação #2: A acurácia do modelo global se mantém estável até níveis moderados de ruído.

Esta reivindicação é validada pela análise da acurácia do modelo global em diferentes níveis de ruído.

*   **Comando:** `python organize_results_accuracy.py`
*   **Tempo esperado:** Menos de 1 minuto.
*   **Resultado esperado:** A geração do gráfico `Acurácia por Ruido.png`. Este gráfico mostra que a acurácia do modelo sofre uma queda pequena e gradual, mantendo-se alta mesmo com níveis de ruído que já são eficazes para mitigar os ataques (conforme a Reivindicação #1).

# LICENSE
