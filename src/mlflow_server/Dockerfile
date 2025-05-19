# Imagem base com CUDA e cuDNN (Ubuntu 20.04)
FROM python:3.10-slim

# Instala dependências do sistema e Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    apache2-utils \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de dependências e instala as bibliotecas necessárias
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copia todo o código fonte para dentro do container
COPY . .

# Create auth directory and setup authentication
RUN mkdir -p auth && \
    htpasswd -cb auth/.htpasswd teacher mlet-password

EXPOSE 5001

# Comando para iniciar a aplicação Flask com autenticação
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////storage/mlflow_logs/mlflow.db", \
     "--default-artifact-root", "file:////storage/training_artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5001", \
     "--basic-auth", "teacher:mlet-password"]
