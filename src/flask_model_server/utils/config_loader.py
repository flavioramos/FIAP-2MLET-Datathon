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
    config = configparser.ConfigParser()
    params_path = os.path.join(PARAMS_DIR, 'params.txt')
    config.read(params_path)

    params = {}

    # Model
    params['MODEL_VERSION'] = int(config.get('Model', 'MODEL_VERSION', fallback='0'))

    # Training
    params['TEST_SIZE'] = float(config.get('Training', 'TEST_SIZE'))
    params['RANDOM_STATE'] = int(config.get('Training', 'RANDOM_STATE'))

    # TFIDF
    params['TFIDF_JOB_DESCRIPTION_MAX_FEATURES'] = int(config.get('TFIDF', 'TFIDF_JOB_DESCRIPTION_MAX_FEATURES'))
    params['TFIDF_JOB_DESCRIPTION_NGRAM_RANGE'] = eval(config.get('TFIDF', 'TFIDF_JOB_DESCRIPTION_NGRAM_RANGE'))
    params['TFIDF_JOB_REQUIREMENTS_MAX_FEATURES'] = int(config.get('TFIDF', 'TFIDF_JOB_REQUIREMENTS_MAX_FEATURES'))
    params['TFIDF_JOB_REQUIREMENTS_NGRAM_RANGE'] = eval(config.get('TFIDF', 'TFIDF_JOB_REQUIREMENTS_NGRAM_RANGE'))
    params['TFIDF_CANDIDATE_CV_MAX_FEATURES'] = int(config.get('TFIDF', 'TFIDF_CANDIDATE_CV_MAX_FEATURES'))
    params['TFIDF_CANDIDATE_CV_NGRAM_RANGE'] = eval(config.get('TFIDF', 'TFIDF_CANDIDATE_CV_NGRAM_RANGE'))
    params['TFIDF_MIN_DF'] = int(config.get('TFIDF', 'TFIDF_MIN_DF'))
    params['TFIDF_MAX_DF'] = float(config.get('TFIDF', 'TFIDF_MAX_DF'))

    # LogisticRegression
    params['LOGISTIC_REGRESSION_MAX_ITER'] = int(config.get('LogisticRegression', 'LOGISTIC_REGRESSION_MAX_ITER'))
    params['LOGISTIC_REGRESSION_TOL'] = float(config.get('LogisticRegression', 'LOGISTIC_REGRESSION_TOL'))

    # GridSearch
    params['GRID_SEARCH_CV'] = int(config.get('GridSearch', 'GRID_SEARCH_CV'))
    params['GRID_SEARCH_SCORING'] = config.get('GridSearch', 'GRID_SEARCH_SCORING')
    params['GRID_SEARCH_N_JOBS'] = int(config.get('GridSearch', 'GRID_SEARCH_N_JOBS'))
    params['GRID_SEARCH_C_VALUES'] = eval(config.get('GridSearch', 'GRID_SEARCH_C_VALUES'))

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
