#!/usr/bin/env python3
"""
Create test users for authentication testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.db import SessionLocal, User
from server.auth import get_password_hash
from datetime import datetime

def create_test_users():
    """Create test users for authentication"""
    db = SessionLocal()
    
    try:
        # Test users data
        test_users = [
            {
                "username": "student1",
                "email": "student1@thothcraft.com",
                "password": "password123",
                "phone_number": None
            },
            {
                "username": "researcher1", 
                "email": "researcher1@thothcraft.com",
                "password": "password123",
                "phone_number": None
            },
            {
                "username": "admin",
                "email": "admin@thothcraft.com", 
                "password": "admin123",
                "phone_number": None
            },
            {
                "username": "testuser",
                "email": "test@thothcraft.com",
                "password": "test123",
                "phone_number": None
            }
        ]
        
        for user_data in test_users:
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.username == user_data["username"]) | 
                (User.email == user_data["email"])
            ).first()
            
            if existing_user:
                print(f"User {user_data['username']} already exists, skipping...")
                continue
            
            # Create new user
            new_user = User(
                username=user_data["username"],
                email=user_data["email"],
                hashed_password=get_password_hash(user_data["password"]),
                phone_number=user_data["phone_number"],
                created_at=datetime.utcnow()
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            print(f"✅ Created user: {user_data['username']} (ID: {new_user.userId})")
        
        print("\n🎉 Test users created successfully!")
        print("\nTest credentials:")
        print("- student1 / password123")
        print("- researcher1 / password123") 
        print("- admin / admin123")
        print("- testuser / test123")
        
    except Exception as e:
        print(f"❌ Error creating test users: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_test_users()
