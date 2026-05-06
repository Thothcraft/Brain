#!/usr/bin/env python3
"""Create gad user for testing."""

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

def create_gad_user():
    """Create gad user if it doesn't exist."""
    db = next(get_db())
    try:
        existing = db.query(User).filter(User.username == 'gad').first()
        if existing:
            print("✅ gad user already exists")
            print(f"   Username: {existing.username}")
            print(f"   User ID: {existing.userId}")
            print(f"   Role: {existing.role}")
            return
        
        gad_user = User(
            username="gad",
            hashed_password=get_password_hash("password"),
            role=0,  # regular user
            plan="free"
        )
        db.add(gad_user)
        db.commit()
        
        print("✅ Created gad user:")
        print(f"   Username: gad")
        print(f"   Password: password")
        print(f"   Role: 0 (regular user)")
        print(f"   Plan: free")
        
    except Exception as e:
        print(f"❌ Error creating gad user: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_gad_user()
