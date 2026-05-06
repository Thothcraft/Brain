#!/usr/bin/env python3
"""Test role-based access control."""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_role_access():
    """Test different user roles accessing protected endpoints."""
    
    print("Testing role-based access control...")
    print("=" * 50)
    
    # Test different users
    users = [
        {"username": "testuser", "password": "password123", "expected_role": 0, "name": "Regular User"},
        {"username": "testorg", "password": "password123", "expected_role": 2, "name": "Organization"},
        {"username": "admin", "password": "password", "expected_role": 1, "name": "Admin"},
    ]
    
    for user in users:
        print(f"\nTesting {user['name']} ({user['username']})...")
        
        # Login
        login_response = requests.post(
            f"{BASE_URL}/api/token",
            data={"username": user["username"], "password": user["password"]}
        )
        
        if login_response.status_code != 200:
            print(f"  ❌ Login failed: {login_response.status_code}")
            continue
        
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Verify role from login response
        actual_role = login_response.json().get("role")
        if actual_role == user["expected_role"]:
            print(f"  ✅ Role correct: {actual_role}")
        else:
            print(f"  ❌ Role mismatch: expected {user['expected_role']}, got {actual_role}")
        
        # Test admin endpoints
        print(f"  Testing admin access...")
        admin_stats = requests.get(f"{BASE_URL}/api/admin/stats", headers=headers)
        if user["expected_role"] == 1:
            if admin_stats.status_code == 200:
                print(f"    ✅ Can access admin stats")
            else:
                print(f"    ❌ Cannot access admin stats: {admin_stats.status_code}")
        else:
            if admin_stats.status_code == 403:
                print(f"    ✅ Correctly blocked from admin stats")
            else:
                print(f"    ❌ Should be blocked but got: {admin_stats.status_code}")
        
        # Test organization endpoints
        print(f"  Testing organization access...")
        org_members = requests.get(f"{BASE_URL}/api/org/members", headers=headers)
        if user["expected_role"] == 2:
            if org_members.status_code == 200:
                print(f"    ✅ Can access org members")
            else:
                print(f"    ❌ Cannot access org members: {org_members.status_code}")
        else:
            if org_members.status_code in [403, 404]:
                print(f"    ✅ Correctly blocked from org members")
            else:
                print(f"    ❌ Should be blocked but got: {org_members.status_code}")
        
        # Test labs endpoints
        print(f"  Testing labs access...")
        labs = requests.get(f"{BASE_URL}/api/labs", headers=headers)
        if user["expected_role"] in [1, 2]:
            if labs.status_code == 200:
                print(f"    ✅ Can access labs")
            else:
                print(f"    ❌ Cannot access labs: {labs.status_code}")
        else:
            if labs.status_code == 403:
                print(f"    ✅ Correctly blocked from labs")
            else:
                print(f"    ❌ Should be blocked but got: {labs.status_code}")

if __name__ == "__main__":
    print("Make sure the backend is running on http://localhost:8000")
    print("Press Enter to continue or Ctrl+C to cancel...")
    input()
    
    test_role_access()
