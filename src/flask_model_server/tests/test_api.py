import pytest
import json
import configparser
import threading
import time

# Load configuration
config = configparser.ConfigParser()
config.read('default_params.txt')

def test_login_success(client):
    """Test successful login."""
    response = client.post("/login", json={
        "username": config.get('Authentication', 'DEFAULT_USERNAME'),
        "password": config.get('Authentication', 'DEFAULT_PASSWORD')
    })
    assert response.status_code == 200
    assert "access_token" in response.json

def test_login_missing_data(client):
    """Test login with missing data."""
    response = client.post("/login", json={})
    assert response.status_code == 400
    assert "error" in response.json

def test_login_invalid_credentials(client):
    """Test login with invalid credentials."""
    response = client.post("/login", json={
        "username": "wrong",
        "password": "wrong"
    })
    assert response.status_code == 401
    assert "error" in response.json

def test_train_endpoint_unauthorized(client):
    """Test train endpoint without authentication."""
    response = client.get("/train")
    assert response.status_code == 401

def test_train_endpoint_authorized(client, auth_headers):
    """Test train endpoint with authentication."""
    response = client.get("/train", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json, dict)

def test_predict_endpoint_unauthorized(client):
    """Test predict endpoint without authentication."""
    response = client.post("/predict", json={
        "job_description": "test job",
        "candidate_cv": "test cv"
    })
    assert response.status_code == 401

def test_predict_endpoint_missing_data(client, auth_headers):
    """Test predict endpoint with missing data."""
    response = client.post("/predict", json={}, headers=auth_headers)
    assert response.status_code == 400
    assert "error" in response.json

def test_predict_endpoint_success(client, auth_headers):
    """Test predict endpoint with valid data."""
    test_data = {
        "principais_atividades": "Python developer with 5 years experience",
        "competencia_tecnicas_e_comportamentais": "Experienced Python developer with ML background",
        "cv_pt": "Experienced Python developer with ML background"
    }
    response = client.post("/predict", json=test_data, headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json, dict)

def test_parallel_training_calls(client, auth_headers):
    """Test parallel calls to training endpoint."""
    def make_training_request():
        response = client.get("/train", headers=auth_headers)
        return response.status_code

    # Start first training request
    thread1 = threading.Thread(target=make_training_request)
    thread1.start()
    
    # Wait a bit to ensure first request has started
    time.sleep(0.5)
    
    # Make second request while first is still running
    response = client.get("/train", headers=auth_headers)
    assert response.status_code == 409  # Should get conflict status
    assert "error" in response.json
    assert "already in progress" in response.json["error"].lower()
    
    # Wait for first request to complete
    thread1.join()
    
    # Verify we can make another request after the first one completes
    response = client.get("/train", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json, dict) 