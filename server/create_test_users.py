#!/usr/bin/env python3
"""Create test users with different roles."""

import os
import sys
from pathlib import Path

# Add server root to path
server_root = Path(__file__).parent
sys.path.insert(0, str(server_root))

from db import get_db, User
import bcrypt

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    if len(password) > 72:
        password = password[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def create_test_users():
    """Create test users with different roles."""
    print("Creating test users...")
    db = next(get_db())
    
    try:
        # Regular user
        regular_user = User(
            username="testuser",
            hashed_password=get_password_hash("password123"),
            role=0,  # regular user
            plan="free"
        )
        db.add(regular_user)
        
        # Organization user
        org_user = User(
            username="testorg",
            hashed_password=get_password_hash("password123"),
            role=2,  # organization
            plan="organization",
            org_name="Test Organization"
        )
        db.add(org_user)
        
        db.commit()
        
        print("✅ Created test users:")
        print(f"   Regular user: testuser / password123 (role=0)")
        print(f"   Organization user: testorg / password123 (role=2)")
        print(f"   Admin user: admin / password (role=1)")
        
    except Exception as e:
        print(f"❌ Error creating users: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_test_users()
