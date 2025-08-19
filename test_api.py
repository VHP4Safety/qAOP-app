#!/usr/bin/env python3
"""
Simple test script for the qAOP API endpoints.
Run this after starting the Flask application.
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_models_endpoint():
    """Test the models information endpoint."""
    print("\nTesting models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/models")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction_invitro():
    """Test invitro model prediction."""
    print("\nTesting invitro prediction...")
    payload = {
        "model_type": "invitro",
        "dose": 10.0,
        "time": 24.0,
        "simulation_count": 10  # Small count for faster testing
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Model: {data['model_type']}")
            print(f"Dose: {data['dose']} {data['dose_unit']}")
            print(f"Time: {data['time']} {data['time_unit']}")
            print(f"DNA Damage (mean): {data['endpoints']['dna_damage']['final_value']['mean']:.4f}")
            print(f"Necrosis (mean): {data['endpoints']['necrosis']['final_value']['mean']:.4f}")
        else:
            print(f"Error response: {response.json()}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction_invivo():
    """Test invivo model prediction."""
    print("\nTesting invivo prediction...")
    payload = {
        "model_type": "invivo",
        "dose": 5.0,
        "time": 48.0,
        "simulation_count": 10  # Small count for faster testing
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Model: {data['model_type']}")
            print(f"Dose: {data['dose']} {data['dose_unit']}")
            print(f"Time: {data['time']} {data['time_unit']}")
            print(f"DNA Damage (mean): {data['endpoints']['dna_damage']['final_value']['mean']:.4f}")
            print(f"Necrosis (mean): {data['endpoints']['necrosis']['final_value']['mean']:.4f}")
            print(f"Inflammation (mean): {data['endpoints']['inflammation']['final_value']['mean']:.4f}")
            print(f"Kidney Failure (mean): {data['endpoints']['kidney_failure']['final_value']['mean']:.4f}")
        else:
            print(f"Error response: {response.json()}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_error_handling():
    """Test API error handling."""
    print("\nTesting error handling...")
    
    # Test missing fields
    payload = {"model_type": "invitro"}  # Missing dose and time
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Missing fields - Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test invalid model type
        payload = {"model_type": "invalid", "dose": 10, "time": 24}
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Invalid model - Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("qAOP API Test Script")
    print("=" * 40)
    print("Make sure the Flask app is running on localhost:5000")
    print("Run with: python app.py")
    print()
    
    tests = [
        test_health_check,
        test_models_endpoint,
        test_prediction_invitro,
        test_prediction_invivo,
        test_error_handling
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ All API tests passed!")
    else:
        print("❌ Some tests failed. Check the Flask app is running.")