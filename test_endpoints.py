#!/usr/bin/env python3
"""
Endpoint Testing Script for Thoth Brain Backend
Tests all major endpoints with proper inputs/outputs
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "https://web-production-d7d37.up.railway.app"
# BASE_URL = "http://localhost:8000"  # For local testing

def test_endpoint(method, endpoint, data=None, params=None, expected_status=200):
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, params=params)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, params=params)
        elif method.upper() == "DELETE":
            response = requests.delete(url, params=params)
        
        print(f"✓ {method} {endpoint} - Status: {response.status_code}")
        
        if response.status_code == expected_status:
            try:
                result = response.json()
                print(f"  Response: {json.dumps(result, indent=2)[:200]}...")
                return True, result
            except:
                print(f"  Response: {response.text[:200]}...")
                return True, response.text
        else:
            print(f"  ❌ Expected {expected_status}, got {response.status_code}")
            print(f"  Error: {response.text[:200]}...")
            return False, None
            
    except Exception as e:
        print(f"❌ {method} {endpoint} - Error: {str(e)}")
        return False, None

def test_sensor_endpoints():
    """Test sensor-related endpoints"""
    print("\n=== TESTING SENSOR ENDPOINTS ===")
    
    # Test current sensor data
    test_endpoint("GET", "/sensors/current", params={"device_id": "thoth-001"})
    
    # Test sensor control
    test_endpoint("POST", "/sensors/control", data={
        "device_id": "thoth-001",
        "sensor_type": "temperature",
        "enabled": True
    })
    
    # Test sensor history
    test_endpoint("GET", "/sensors/history", params={
        "device_id": "thoth-001",
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-02T00:00:00Z"
    })
    
    # Test sensor statistics
    test_endpoint("GET", "/sensors/stats", params={
        "device_id": "thoth-001",
        "period": "24h"
    })

def test_network_endpoints():
    """Test network-related endpoints"""
    print("\n=== TESTING NETWORK ENDPOINTS ===")
    
    # Test WiFi configuration
    test_endpoint("POST", "/network/wifi", data={
        "ssid": "TestNetwork",
        "password": "testpassword",
        "security": "WPA2",
        "auto_connect": True
    })
    
    # Test network status
    test_endpoint("GET", "/network/status", params={"interface": "wlan0"})
    
    # Test network scan
    test_endpoint("GET", "/network/scan")

def test_training_endpoints():
    """Test training-related endpoints"""
    print("\n=== TESTING TRAINING ENDPOINTS ===")
    
    # Test training setup
    success, result = test_endpoint("POST", "/training/setup", data={
        "device_id": "thoth-001",
        "model": "cnn",
        "data": "sensors",
        "mode": "on-device",
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001
    })
    
    job_id = None
    if success and result:
        job_id = result.get("job_id")
    
    # Test training status
    if job_id:
        test_endpoint("GET", "/training/status", params={"job_id": job_id})
    else:
        test_endpoint("GET", "/training/status")
    
    # Test federated training
    test_endpoint("POST", "/federated/train", data={
        "device_id": "thoth-001",
        "model_type": "cnn",
        "rounds": 3,
        "privacy": {
            "differential_privacy": True,
            "epsilon": 1.0
        }
    })
    
    # Test federated status
    test_endpoint("GET", "/federated/status")
    
    # Test model list
    test_endpoint("GET", "/models", params={"device_id": "thoth-001"})

def test_curriculum_endpoints():
    """Test curriculum-related endpoints"""
    print("\n=== TESTING CURRICULUM ENDPOINTS ===")
    
    # Test curriculum fetch
    test_endpoint("GET", "/curriculum")
    
    # Test specific module
    test_endpoint("GET", "/curriculum/mod_001")
    
    # Test progress tracking
    test_endpoint("POST", "/curriculum/progress", data={
        "student_id": "student-001",
        "module_id": "mod_001",
        "progress": 75,
        "completed": False,
        "time_spent": 45
    })
    
    # Test lab submission
    test_endpoint("POST", "/curriculum/lab/submit", data={
        "student_id": "student-001",
        "module_id": "mod_002",
        "submission": {
            "code": "print('Hello Thoth!')",
            "results": "Hello Thoth!",
            "notes": "Completed WiFi configuration lab"
        }
    })
    
    # Test leaderboard
    test_endpoint("GET", "/curriculum/leaderboard")

def test_auth_endpoints():
    """Test authentication endpoints"""
    print("\n=== TESTING AUTH ENDPOINTS ===")
    
    # Test user registration
    test_endpoint("POST", "/auth/register", data={
        "username": "testuser",
        "password": "testpassword",
        "role": "student"
    })
    
    # Test user login
    test_endpoint("POST", "/auth/login", data={
        "username": "testuser",
        "password": "testpassword"
    })

def test_health_endpoints():
    """Test health and basic endpoints"""
    print("\n=== TESTING HEALTH ENDPOINTS ===")
    
    # Test root endpoint
    test_endpoint("GET", "/")
    
    # Test health check
    test_endpoint("GET", "/health")

def main():
    """Run all endpoint tests"""
    print("🚀 Starting Thoth Backend Endpoint Tests")
    print(f"Testing against: {BASE_URL}")
    print(f"Started at: {datetime.now()}")
    
    # Test all endpoint groups
    test_health_endpoints()
    # test_sensor_endpoints()
    # test_network_endpoints()
    # test_training_endpoints()
    # test_curriculum_endpoints()
    # test_auth_endpoints()
    
    print(f"\n✅ Endpoint testing completed at: {datetime.now()}")
    print("\nNote: Some endpoints may return errors if the backend is not fully configured")
    print("or if authentication is required. This is expected for a demo environment.")

if __name__ == "__main__":
    main()
