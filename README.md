# Título do projeto

**Título do Artigo:** Avaliação da Privacidade Diferencial Aplicada ao Aprendizado Federado Através de Ataques de Inversão de Modelo

**Resumo:** Este trabalho propõe uma metodologia prática para avaliar a eficácia de mecanismos de Privacidade Diferencial (DP) no Aprendizado Federado (FL) diante de ataques de inversão de modelo. Para isso, foi adotado um cenário adversarial baseado no paradigma de segurança cibernética equipe vermelha/equipe azul (RT/BT), onde a equipe azul implementa a proteção de DP e a equipe vermelha realiza ataques para recuperar dados de um cliente específico a partir do modelo global. Foram implementados dois tipos de ataques — um baseado em gradientes e outro ingênuo — aplicados em diferentes intensidades de ruído gaussiano. Os experimentos demonstraram que ruídos baixos já são suficientes para mitigar significativamente os ataques de inversão, conforme evidenciado tanto por métricas quantitativas (SSIM, PSNR e MSE) quanto pela análise visual das imagens reconstruídas. Por outro lado, observou-se que a acurácia do modelo global se mantém estável até níveis moderados de ruído, com impacto limitado no desempenho. Esses resultados indicam que é possível alcançar um bom equilíbrio entre privacidade e utilidade do modelo em cenários práticos.

# Estrutura do readme.md

Este documento está organizado para guiar os avaliadores desde a configuração do ambiente até a replicação completa dos resultados do artigo. As seções incluem:

*   **Título e Resumo:** Apresenta o contexto do artefato.
*   **Selos Considerados e Preocupações com Segurança:** Informações importantes para o processo de avaliação.
*   **Informações básicas, Dependências e Instalação:** Guias para preparar o ambiente de execução, incluindo opções para Docker.
*   **Teste mínimo:** Um passo rápido para validar a instalação do ambiente.
*   **Experimentos:** Instruções detalhadas para reproduzir cada reivindicação do artigo, com opções de fluxo automatizado e manual.
*   **LICENSE:** Informações sobre a licença do software.

# Selos Considerados

Os selos considerados para este artefato são: **Disponível (SELOD)**, **Funcional (SELOF)**, **Sustentável (SELOS)** e **Reprodutível (SELOR)**.

# Informações básicas

Esta seção descreve os requisitos de hardware e software para a execução dos experimentos.

*   **Hardware:**
    *   Computador padrão com pelo menos **8 GB de RAM**.
    *   Aproximadamente **200 MB** de espaço livre em disco para o código, dependências e resultados.
*   **Software:**
    *   Sistema Operacional: Windows, macOS ou Linux.
    *   **Opção Local:** Python 3.11+ e Git.
    *   **Opção Docker:** Docker Desktop (Windows/macOS) ou Docker Engine (Linux).

# Dependências

As dependências de software e dados para a execução estão listadas abaixo.

# Preocupações com segurança

A execução deste artefato é segura. Os scripts realizam apenas cálculos numéricos e manipulação de arquivos dentro do diretório do projeto, sem acessar informações sensíveis do sistema ou da rede. Um download do conjunto de dados MNIST pode ser realizado automaticamente do site `python-course.eu` via **HTTPS**, caso o arquivo não seja encontrado localmente.

### Bibliotecas Python
As bibliotecas necessárias, com suas versões exatas para garantir a reprodutibilidade, estão listadas no arquivo `requirements.txt`.
*   `numpy==1.26.4`
*   `pandas==2.2.2`
*   `matplotlib==3.9.0`
*   `scikit-image==0.25.2`
*   `scipy==1.13.1`

### Conjunto de Dados
Este projeto utiliza o conjunto de dados **MNIST** em formato CSV.

> **✨ Automação:** Não é necessário baixar o dataset manualmente. O script principal (`MNIST_FL_inversion_comparison.py`) verificará se o arquivo `mnist_train.csv` existe no diretório raiz e, caso contrário, fará o download automaticamente de uma fonte segura.

# Instalação

Escolha uma das opções abaixo para configurar seu ambiente. A opção Docker é recomendada para máxima reprodutibilidade.

### Opção 1: Instalação Local
1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2.  **Crie e ative um ambiente virtual (Recomendado):**
    ```bash
    # Para macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # Para Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

### Opção 2: Usando Docker (Recomendado para Reprodutibilidade)
1.  **Pré-requisito:** Certifique-se de que o Docker está instalado e em execução.

2.  **Construa a imagem Docker:** Na raiz do projeto, execute o comando abaixo.
    ```bash
    docker build -t fl-privacy-eval .
    ```
Ao final de qualquer uma das opções, o ambiente estará pronto para execução.

# Teste mínimo

Execute um teste rápido para garantir que a configuração está correta. O script principal, quando executado sem argumentos, roda uma configuração mínima (1 classe, 1 nível de ruído, 1 seed), ideal para verificação.

*   **Execução Local:**
    ```bash
    python MNIST_FL_inversion_comparison.py
    ```

*   **Execução com Docker:**
    ```bash
    # Para macOS/Linux
    docker run --rm -it -v "$(pwd):/app" fl-privacy-eval

    # Para Windows (PowerShell)
    docker run --rm -it -v "${PWD}:/app" fl-privacy-eval
    ```
    > **Nota:** O comando `-v "$(pwd):/app"` mapeia o diretório atual do seu computador para o diretório `/app` dentro do contêiner. Isso é **essencial** para que os resultados (arquivos CSV, gráficos) sejam salvos na sua máquina.

*   **Tempo esperado:** 2-3 minutos.
*   **Resultado esperado:** O script deve finalizar sem erros, imprimir uma mensagem de conclusão e gerar o arquivo `new_results_with_seed.csv` junto com as pastas de imagens reconstruídas.

# Experimentos

Oferecemos duas maneiras de executar os experimentos completos para replicar os resultados do artigo: um fluxo automatizado e um fluxo manual passo a passo.

### Opção A: Fluxo de Trabalho Completo (Automatizado)

O script `run_all_tests.py` orquestra a execução do experimento principal e de todos os scripts de análise em sequência, gerando todos os gráficos e resultados de uma só vez.

*   **Execução Local:**
    ```bash
    python run_all_tests.py
    ```

*   **Execução com Docker:**
    ```bash
    # Para macOS/Linux
    docker run --rm -it -v "$(pwd):/app" fl-privacy-eval python run_all_tests.py
    
    # Para Windows (PowerShell)
    docker run --rm -it -v "${PWD}:/app" fl-privacy-eval python run_all_tests.py
    ```
*   **Tempo esperado de execução:** Aproximadamente **45 a 60 horas** (dependendo da configuração de seeds; o padrão do script é 30 seeds).

### Opção B: Execução Manual Passo a Passo

Esta abordagem oferece mais controle, permitindo executar cada etapa individualmente e analisar os resultados intermediários.

#### Passo 1: Geração dos Dados Brutos

Execute o script principal com parâmetros para gerar todos os dados brutos. Para uma análise estatística robusta, recomenda-se executar com 30 `seeds`.

*   **Comando Local:**
    ```bash
    python MNIST_FL_inversion_comparison.py --number_of_seeds 30 --classes "0,1,2,3,4,5,6,7,8,9" --noise_scales "0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1"
    ```
*   **Comando com Docker:**
    ```bash
    # Para macOS/Linux
    docker run --rm -it -v "$(pwd):/app" fl-privacy-eval python MNIST_FL_inversion_comparison.py --number_of_seeds 30 --classes "0,1,2,3,4,5,6,7,8,9" --noise_scales "0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1"
    ```
*   **Tempo esperado:** ~45-60 horas.
*   **Resultado esperado:** Geração do arquivo `new_results_with_seed.csv` e pastas com as imagens reconstruídas.

> **Nota sobre Tratamento de Erros:**
> Devido à distribuição aleatória de dados entre os clientes, pode ocorrer de um cliente específico (o alvo do ataque) não possuir exemplos para uma determinada classe. Ao tentar realizar um ataque para essa classe, o script encontraria um erro. Um bloco `try...except` foi implementado para capturar essa exceção, descartar os resultados parciais da combinação `seed/classe` que falhou e continuar a execução para a próxima classe, garantindo a robustez do processo de longa duração. Assim, a execução termina quando a última classe alcançar o número alvo de seeds para qual foi testada.

---

#### Passo 2: Validação da Reivindicação #1
**"Ruídos baixos mitigam significativamente os ataques de inversão."**

Esta reivindicação é validada pela análise de métricas de qualidade de imagem (SSIM, MSE e PSNR). O script abaixo gera os gráficos que demonstram a degradação da qualidade das imagens reconstruídas com o aumento do ruído.

*   **Execução (Local ou Docker):**
    ```bash
    # Local
    python organize_attack_results.py

    # Docker (Exemplo macOS/Linux)
    docker run --rm -it -v "$(pwd):/app" fl-privacy-eval python organize_attack_results.py
    ```
*   **Resultado esperado:** Gráficos para SSIM, MSE e PSNR salvos na pasta `graficos/`, mostrando visualmente a eficácia da mitigação.

---

#### Passo 3: Validação da Reivindicação #2
**"A acurácia do modelo global se mantém estável até níveis moderados de ruído."**

Esta reivindicação é validada pela análise da acurácia do modelo global em diferentes níveis de ruído. O script a seguir gera o gráfico correspondente.

*   **Execução (Local ou Docker):**
    ```bash
    # Local
    python organize_results_accuracy.py

    # Docker (Exemplo macOS/Linux)
    docker run --rm -it -v "$(pwd):/app" fl-privacy-eval python organize_results_accuracy.py
    ```
*   **Resultado esperado:** O gráfico `Acurácia por Ruido.png` na pasta `graficos/`, ilustrando que a acurácia sofre uma queda gradual e limitada.


---

#### Passo 4: Análise de Significância Estatística

Para validar estatisticamente as observações, foi utilizado o teste de Mann-Whitney U. O script `Mann_Whitney.py` realiza essa análise comparando os resultados com e sem ruído.

*   **Execução (Local ou Docker) - Exemplo com ruído 0.07:**
    ```bash
    # Local
    python Mann_Whitney.py --selected_noise 0.07

    # Docker (Exemplo macOS/Linux)
    docker run --rm -it -v "$(pwd):/app" fl-privacy-eval python Mann_Whitney.py --selected_noise 0.07
    ```
*   **Resultado esperado:** O script imprimirá no terminal o p-valor para as métricas de ataque (SSIM, MSE, PSNR) e para a acurácia, comparando o cenário sem ruído com o nível de ruído selecionado. Os testes confirmam que:
    1.  As métricas de qualidade de imagem com ruído são **estatisticamente diferentes** daquelas sem ruído, mesmo para os menores níveis.
    2.  A acurácia do modelo só apresenta uma diferença **estatisticamente significativa** a partir de níveis de ruído moderados (ex: 0.07), apoiando a conclusão de que é possível adicionar ruído para privacidade sem impactar significativamente a utilidade do modelo.


# LICENSE

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo `LICENSE` para obter mais detalhes.
