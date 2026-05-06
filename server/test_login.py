#!/usr/bin/env python3
"""Test admin login."""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_admin_login():
    """Test admin login."""
    print("Testing admin login...")
    
    # Test admin login
    response = requests.post(
        f"{BASE_URL}/api/token",
        data={"username": "admin", "password": "password"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Login successful!")
        print(f"   Access Token: {data.get('access_token', 'N/A')[:50]}...")
        print(f"   User ID: {data.get('user_id')}")
        print(f"   Role: {data.get('role')}")
        print(f"   Plan: {data.get('plan')}")
    else:
        print("❌ Login failed!")
        if response.headers.get('content-type', '').startswith('application/json'):
            print(f"   Error: {response.json()}")

if __name__ == "__main__":
    print("Make sure the backend is running on http://localhost:8000")
    print("Press Enter to test admin login...")
    input()
    
    test_admin_login()
