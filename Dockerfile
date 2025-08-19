# --- Estágio 1: Definir a imagem base ---
# Usamos uma imagem oficial do Python 3.11. A versão 'slim' é mais leve e suficiente.
FROM python:3.11-slim

# --- Estágio 2: Configurar o ambiente ---
# Definir o diretório de trabalho dentro do contêiner.
WORKDIR /app

# --- Estágio 3: Instalar as dependências Python ---
# Copiar apenas o arquivo de requisitos primeiro para aproveitar o cache do Docker.
# As dependências só serão reinstaladas se este arquivo mudar.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Estágio 4: Definir o comando de execução ---
# Comando padrão que será executado quando o contêiner iniciar.
CMD ["python", "MNIST_FL_inversion_comparison.py"]