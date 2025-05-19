"""Utility functions for handling data files and operations."""

import os
import gdown
import zipfile
import configparser
from pathlib import Path


def download_and_extract_data(data_dir: str, params_dir: str) -> None:
    """Download and extract data if not already present.
    
    Args:
        data_dir: Directory where data files should be stored
        params_dir: Directory containing the params.txt file
    """
    # Read DATA_GDRIVE_FILE_ID from params.txt
    config = configparser.ConfigParser()
    params_file = os.path.join(params_dir, "params.txt")
    
    if not os.path.exists(params_file):
        print(f"Warning: {params_file} not found. Using default file ID.")
        file_id = "14RDmgVVHg6limnuS14PXaI3lHVP_gd1z"  # Default file ID
    else:
        try:
            config.read(params_file)
            file_id = config.get('Training', 'DATA_GDRIVE_FILE_ID')
        except (configparser.Error, KeyError) as e:
            print(f"Warning: Error reading {params_file}: {str(e)}. Using default file ID.")
            file_id = "14RDmgVVHg6limnuS14PXaI3lHVP_gd1z"  # Default file ID
    
    # Check if data files exist
    required_files = ['applicants.json', 'vagas.json', 'prospects.json']
    data_files_exist = all(
        os.path.exists(os.path.join(data_dir, file))
        for file in required_files
    )
    
    if not data_files_exist:
        print("Data files not found. Downloading and extracting...")
        try:
            # Download the zip file using gdown
            zip_path = os.path.join(data_dir, "temp_data.zip")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, zip_path, quiet=False)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove the temporary zip file
            os.remove(zip_path)
            print("Data downloaded and extracted successfully.")
        except Exception as e:
            print(f"Error downloading or extracting data: {str(e)}")
            raise
    else:
        print("Data files already exist.")


def ensure_data_directory(data_dir: str) -> None:
    """Ensure the data directory exists.
    
    Args:
        data_dir: Directory where data files should be stored
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")


def get_data_file_paths(data_dir: str) -> dict:
    """Get the paths for all data files.
    
    Args:
        data_dir: Directory where data files are stored
        
    Returns:
        Dictionary containing paths to all data files
    """
    return {
        'applicants': os.path.join(data_dir, "applicants.json"),
        'vagas': os.path.join(data_dir, "vagas.json"),
        'prospects': os.path.join(data_dir, "prospects.json")
    } 