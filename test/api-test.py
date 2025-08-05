import requests
import json
import pandas as pd

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("=== HEALTH CHECK ===")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}\n")

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/model/info")
    print("=== MODEL INFO ===")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}\n")

def test_single_prediction():
    """Test single prediction"""
    # Example loan data - adjust based on your actual features
    try:
        temp = pd.read_csv('data/auto_test_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: 'data/auto_test_data.csv' not found. Please ensure the file exists.")
        return
    
    # get the 10 row of the data
    sample_data = temp.iloc[9].to_dict()
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    print("=== SINGLE PREDICTION ===")
    print("Input:", json.dumps(sample_data, indent=2))
    print("Response:", json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}\n")

def test_batch_prediction():
    """Test batch prediction"""
    # Example batch data
    try:
        batch_data = pd.read_csv('data/auto_test_data.csv', low_memory=False).to_dict(orient='records')
    except FileNotFoundError:
        print("Error: 'data/auto_test_data.csv' not found. Please ensure the file exists.")
        return

    # get the 15-18 row of the data
    batch_data = batch_data[14:18]  # Adjust the slice as needed

    payload = {
        "data": batch_data
    }

    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    print("=== BATCH PREDICTION ===")
    print("Input:", json.dumps(batch_data, indent=2))
    print("Response:", json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}\n")

if __name__ == "__main__":
    print("Testing Loan Risk Prediction API\n")
    
    test_health()
    test_model_info()
    test_single_prediction()
    test_batch_prediction()