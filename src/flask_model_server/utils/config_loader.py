"""Configuration loader module.

This module provides functionality to load parameters from a configuration file.
"""

import os
import configparser
from config import PARAMS_DIR


def load_parameters():
    """Load parameters from the configuration file.
    
    Returns:
        dict: Dictionary containing the loaded parameters
    """
    params = {}
    file_path = f"{PARAMS_DIR}/params.txt"
    print(f"Loading parameters from {file_path}")
    
    if os.path.exists(file_path):
        config = configparser.ConfigParser()
        config.read(file_path)
        
        # Convert config sections to flat dictionary
        for section in config.sections():
            for key, value in config[section].items():
                try:
                    params[key] = int(value)
                except ValueError:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value
    else:
        print(f"File {file_path} not found. Using default parameters.")
    
    return params


def save_parameters(params):
    """Save parameters to the configuration file.
    
    Args:
        params (dict): Dictionary containing the parameters to save
    """
    file_path = f"{PARAMS_DIR}/params.txt"
    print(f"Saving parameters to {file_path}")
    
    # Create config parser
    config = configparser.ConfigParser()
    
    # First load existing parameters if file exists
    if os.path.exists(file_path):
        config.read(file_path)
    
    # Define sections and their parameters
    sections = {
        'Model': ['MODEL_VERSION'],
        'Training': ['DATA_GDRIVE_FILE_ID', 'TEST_SIZE', 'RANDOM_STATE'],
        'TFIDF': [
            'TFIDF_JOB_DESCRIPTION_MAX_FEATURES',
            'TFIDF_JOB_DESCRIPTION_NGRAM_RANGE',
            'TFIDF_JOB_REQUIREMENTS_MAX_FEATURES',
            'TFIDF_JOB_REQUIREMENTS_NGRAM_RANGE',
            'TFIDF_CANDIDATE_CV_MAX_FEATURES',
            'TFIDF_CANDIDATE_CV_NGRAM_RANGE'
        ],
        'LogisticRegression': ['LOGISTIC_REGRESSION_MAX_ITER'],
        'GridSearch': [
            'GRID_SEARCH_CV',
            'GRID_SEARCH_SCORING',
            'GRID_SEARCH_N_JOBS',
            'GRID_SEARCH_C_VALUES'
        ],
        'Authentication': [
            'DEFAULT_USERNAME',
            'DEFAULT_PASSWORD',
            'JWT_SECRET_KEY'
        ]
    }
    
    # Add sections and their parameters
    for section, keys in sections.items():
        if section not in config:
            config[section] = {}
        for key in keys:
            if key in params:
                config[section][key] = str(params[key])
    
    # Write to file
    with open(file_path, 'w') as f:
        config.write(f)
