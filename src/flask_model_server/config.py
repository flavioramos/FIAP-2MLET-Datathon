"""Configuration settings for the ML model training and serving.

This module contains all the configuration parameters, paths, and constants
used throughout the application for both local and containerized environments.
"""

import os
import sys
import shutil
from utils.data_utils import download_and_extract_data, ensure_data_directory, get_data_file_paths


# Environment detection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL = sys.argv[-1] == 'local' or os.getenv('LOCAL_RUN') == 'true'  # Local session or remote (container)

print(f"Running locally: {LOCAL}")


# Directory paths
if LOCAL:
    ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../local_storage/training_artifacts"))
    LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../local_storage/mlflow_logs"))
    PARAMS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../local_storage/params"))
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../local_storage/data"))
else:
    ARTIFACTS_DIR = os.path.abspath(os.path.join("/storage/", "training_artifacts"))
    LOGS_DIR = os.path.abspath(os.path.join("/storage/", "mlflow_logs"))
    PARAMS_DIR = os.path.abspath(os.path.join("/storage/", "params"))
    DATA_DIR = os.path.abspath(os.path.join("/storage/", "data"))


# Create necessary directories
for directory in [ARTIFACTS_DIR, LOGS_DIR, PARAMS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)


# Copy default parameters if needed
if not os.path.exists(os.path.join(PARAMS_DIR, "params.txt")):
    shutil.copyfile(
        os.path.join(BASE_DIR, "default_params.txt"),
        os.path.join(PARAMS_DIR, "params.txt")
    )
    print(f"Default params.txt copied to {PARAMS_DIR}")


# Ensure data directory exists and download data if needed
ensure_data_directory(DATA_DIR)
download_and_extract_data(DATA_DIR, PARAMS_DIR)


# Print directory paths
print(f"ARTIFACTS_DIR: {ARTIFACTS_DIR}")
print(f"LOGS_DIR: {LOGS_DIR}")
print(f"PARAMS_DIR: {PARAMS_DIR}")
print(f"DATA_DIR: {DATA_DIR}")


# Generated file paths
MODEL_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")


# Status mapping
STATUS_MAP = {
    "Encaminhado ao Requisitante":        0,
    "Contratado pela Decision":           1,
    "Desistiu":                           0,
    "Documentação PJ":                    1,
    "Não Aprovado pelo Cliente":          0,
    "Prospect":                           0,
    "Não Aprovado pelo RH":               0,
    "Aprovado":                           1,
    "Não Aprovado pelo Requisitante":     0,
    "Inscrito":                           0,
    "Entrevista Técnica":                 0,
    "Em avaliação pelo RH":               0,
    "Contratado como Hunting":            1,
    "Desistiu da Contratação":            0,
    "Entrevista com Cliente":             0,
    "Documentação CLT":                   1,
    "Recusado":                           0,
    "Documentação Cooperado":             1,
    "Sem interesse nesta vaga":           0,
    "Encaminhar Proposta":                1,
    "Proposta Aceita":                    1
}


# Data paths
data_paths = get_data_file_paths(DATA_DIR)
APPLICANTS_PATH = data_paths['applicants']
VAGAS_PATH = data_paths['vagas']
PROSPECTS_PATH = data_paths['prospects']