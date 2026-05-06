#!/usr/bin/env python3
"""Test Stripe endpoint locally."""

import requests
import json

# Test data
BASE_URL = "http://localhost:8000"
TEST_USER = {
    "username": "testuser",
    "password": "testpass123"
}

def test_stripe_checkout():
    """Test creating a Stripe checkout session."""
    print("Testing Stripe checkout endpoint...")
    
    # First, login to get token
    login_response = requests.post(
        f"{BASE_URL}/api/token",
        data={"username": "admin", "password": "password"}
    )
    
    if login_response.status_code != 200:
        print(f"❌ Login failed: {login_response.status_code}")
        print(login_response.text)
        return
    
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test researcher plan checkout
    plans = ["researcher", "organization"]
    for plan in plans:
        print(f"\nTesting {plan} plan...")
        
        checkout_response = requests.post(
            f"{BASE_URL}/api/stripe/create-checkout-session",
            json={"plan": plan},
            headers=headers
        )
        
        if checkout_response.status_code == 200:
            data = checkout_response.json()
            print(f"✅ {plan} checkout session created")
            print(f"   URL: {data.get('url', 'No URL returned')}")
        else:
            print(f"❌ {plan} checkout failed: {checkout_response.status_code}")
            print(checkout_response.text)

if __name__ == "__main__":
    print("Make sure the backend is running on http://localhost:8000")
    print("Press Enter to continue or Ctrl+C to cancel...")
    input()
    
    test_stripe_checkout()
