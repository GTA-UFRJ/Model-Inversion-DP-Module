# Avaliação da Privacidade Diferencial no Aprendizado Federado

**Título do Artigo:** Avaliação da Privacidade Diferencial Aplicada ao Aprendizado Federado Através de Ataques de Inversão de Modelo

**Resumo:** Este trabalho propõe uma metodologia prática para avaliar a eficácia de mecanismos de Privacidade Diferencial (DP) no Aprendizado Federado (FL) diante de ataques de inversão de modelo. Para isso, foi adotado um cenário adversarial baseado no paradigma de segurança cibernética equipe vermelha/equipe azul (RT/BT), onde a equipe azul implementa a proteção de DP e a equipe vermelha realiza ataques para recuperar dados de um cliente específico a partir do modelo global. Foram implementados dois tipos de ataques — um baseado em gradientes e outro ingênuo — aplicados em diferentes intensidades de ruído gaussiano. Os experimentos demonstraram que ruídos baixos já são suficientes para mitigar significativamente os ataques de inversão, conforme evidenciado tanto por métricas quantitativas (SSIM, PSNR e MSE) quanto pela análise visual das imagens reconstruídas. Por outro lado, observou-se que a acurácia do modelo global se mantém estável até níveis moderados de ruído, com impacto limitado no desempenho. Esses resultados indicam que é possível alcançar um bom equilíbrio entre privacidade e utilidade do modelo em cenários práticos.

# Estrutura do readme.md

Este documento está organizado nas seguintes seções para guiar os avaliadores na configuração do ambiente e na replicação dos resultados do artigo:

*   **Título projeto:** Apresenta o título e o resumo do artigo associado.
*   **Estrutura do readme.md:** Descreve a organização deste documento.
*   **Selos Considerados:** Lista os selos de qualidade almejados na avaliação.
*   **Informações básicas:** Detalha os requisitos de hardware e software para execução.
*   **Dependências:** Lista as bibliotecas Python e o conjunto de dados necessários.
*   **Preocupações com segurança:** Informa sobre possíveis riscos durante a execução.
*   **Instalação:** Apresenta um guia passo a passo para configurar o ambiente de execução.
*   **Teste mínimo:** Fornece um teste rápido para validar a instalação.
*   **Experimentos:** Contém instruções detalhadas para reproduzir as principais reivindicações do artigo.
*   **LICENSE:** Apresenta a licença de software do projeto.

# Selos Considerados

Os selos considerados para este artefato são: **Disponível (SELOD)**, **Funcional (SELOF)**, **Sustentável (SELOS)** e **Reprodutível (SELOR)**.

# Informações básicas

*   **Hardware:** Um computador padrão com pelo menos 8 GB de RAM e 200 MB de espaço em disco.
*   **Software:**
    *   Sistema Operacional: Windows, macOS ou Linux.
    *   Python 3.11 ou superior.

# Dependências

As dependências de software e dados necessários para a execução dos experimentos estão listadas abaixo.

### Bibliotecas Python

As bibliotecas podem ser instaladas via `pip`. As versões exatas usadas no desenvolvimento estão listadas abaixo, mas versões mais recentes ou até um pouco mais antigas provavelmente funcionarão. Está disponibilizado também um arquivo `requirements.txt` para facilitar a instalação.

*   `numpy==1.26.4`
*   `pandas==2.2.2`
*   `matplotlib==3.9.0`
*   `scikit-image==0.25.2`
*   `torch==2.5.1+cu121`
*   `opencv-python==4.11.0.86`
*   `scipy==1.13.1`

### Conjunto de Dados

O conjunto de dados MNIST, no formato CSV, não está incluído neste repositório. Ele pode ser baixado do Kaggle através do seguinte link:

*   [MNIST in CSV no Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

É necessário baixar o arquivo `mnist_train.csv` e colocá-lo no diretório raiz do projeto.

# Preocupações com segurança

A execução deste artefato não apresenta riscos de segurança conhecidos. Os scripts apenas realizam cálculos numéricos, manipulam arquivos dentro do diretório do projeto e não acessam informações sensíveis do sistema ou da rede.

# Instalação

1.  Clone ou baixe este repositório para sua máquina local.
2.  Instale as dependências Python listadas na seção **Dependências** usando o `pip`:
    ```bash
    pip install numpy pandas matplotlib scikit-image torch opencv-python scipy
    ```
3.  Faça o download do arquivo `mnist_train.csv` conforme as instruções na seção **Dependências** e coloque-o na raiz do projeto.

Ao final destes passos, o ambiente estará pronto para execução.

# Teste mínimo

Para garantir que o ambiente está configurado corretamente, execute um teste mínimo que roda o experimento para apenas uma classe, um nível de ruído e por 50 rodadas de treinamento.

1.  Abra o script `MNIST_FL_inversion_comparison.py` em um editor de texto.
2.  No final do arquivo (`if __name__ == "__main__":`), modifique os parâmetros dos laços de repetição conforme o exemplo abaixo:
    ```python
    # Bloco principal para execução
    if __name__ == "__main__":
    for seed in range(1): # Executa para 1 semente
        for i in range(1): # Executa para 1 classe
            ocorreu_erro_neste_i = False # Flag para caso de erros
            for NOISE_SCALEtime100 in range(1): # Executa para 1 nível de ruído
            # ...
    ```
3.  Execute o script a partir do terminal:
    ```bash
    python MNIST_FL_inversion_comparison.py
    ```

A execução deve levar até 1 minuto, mas pode demorar mais dependendo das especificações do dispositivo onde a execução está ocorrendo. Se o script terminar sem erros, imprimir a mensagem `acabou` e gerar o arquivo `new_results_with_seed.csv` na raiz do repositório e imagens de exemplo na pasta `0/0.0 noise/`, a instalação foi bem-sucedida.

**Importante:** Lembre-se de reverter as alterações no script antes de executar os experimentos completos.

# Experimentos

Esta seção descreve como reproduzir os resultados e as principais reivindicações do artigo.

### Passo 1: Execução Completa dos Experimentos

Execute o script principal sem modificações para gerar todos os dados brutos. Para uma análise estatística robusta, recomenda-se executar com pelo menos 30 `seeds` (alterando o parâmetro `range` no laço de repetição principal).

*   **Comando:** `python MNIST_FL_inversion_comparison.py`
*   **Tempo esperado:** 2-3 horas para 2 `seeds`; ~45-60 horas para 30 `seeds`.
*   **Recursos esperados:** ~1 GB de RAM e ~100 MB de espaço em disco para os resultados.
*   **Resultado esperado:** Um arquivo `new_results_with_seed.csv` e uma estrutura de diretórios com as imagens reconstruídas (`<seed>/<noise_level>/<label>_reconstructed.png`).

> **Nota sobre Tratamento de Erros:**
> Devido à distribuição aleatória de dados entre os clientes, pode ocorrer de um cliente específico não possuir exemplos para uma determinada classe. Ao tentar realizar um ataque para essa classe, o script encontrará um erro por falta de imagens de referência. Um bloco `try...except` foi implementado para capturar essa exceção, limpar os resultados parciais da combinação `seed/label` que falhou e continuar a execução para a próxima classe, garantindo a robustez do processo.

### Passo 2: Validação da Reivindicação #1
**"Ruídos baixos mitigam significativamente os ataques de inversão."**

Esta reivindicação é validada pela análise de métricas de qualidade de imagem (MSE, PSNR e SSIM). Execute os seguintes scripts para gerar os gráficos que demonstram a degradação da qualidade das imagens reconstruídas com o aumento do ruído.

*   **Comandos:**
    ```bash
    python organize_results_MSE.py
    python organize_results_PSNR.py
    python organize_results_SSIM.py
    ```
*   **Tempo esperado:** Menos de 1 minuto para cada script.
*   **Resultado esperado:** Os gráficos `*_por_ruido_médio.png` (MSE, PSNR, SSIM), que mostram visualmente a eficácia da mitigação.

### Passo 3: Validação da Reivindicação #2
**"A acurácia do modelo global se mantém estável até níveis moderados de ruído."**

Esta reivindicação é validada pela análise da acurácia do modelo global em diferentes níveis de ruído.

*   **Comando:** `python organize_results_accuracy.py`
*   **Tempo esperado:** Menos de 1 minuto.
*   **Resultado esperado:** O gráfico `Acurácia por Ruido.png`, que ilustra que a acurácia sofre uma queda gradual e limitada, mantendo-se alta mesmo com níveis de ruído que já mitigam os ataques (conforme a Reivindicação #1).

### Passo 4: Análise de Significância Estatística (Opcional)

Para validar estatisticamente as observações das reivindicações anteriores, foi utilizado o teste de Mann-Whitney U. A implementação está disponível no notebook `Mann_Whitney.ipynb`.

*   **Requisitos:** Jupyter Notebook ou Jupyter Lab (`pip install notebook` ou `pip install jupyterlab`).
*   **Execução:** Abra e execute as células do notebook `Mann_Whitney.ipynb`.

Os testes confirmam que:
1.  As métricas de qualidade de imagem (MSE, PSNR, SSIM) com ruído são estatisticamente diferentes daquelas sem ruído, mesmo para os menores níveis.
2.  Com um número suficiente de execuções (pelo menos 30 `seeds`), a acurácia do modelo só apresenta uma diferença estatisticamente significativa a partir de um nível de ruído próximo a 0.07, o que apoia a conclusão de que é possível adicionar ruído sem impactar significativamente a utilidade do modelo.

# LICENSE

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo `LICENSE` para obter mais detalhes.