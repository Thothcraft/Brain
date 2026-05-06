#!/usr/bin/env python3
"""Verify admin password hash."""

import sys
from pathlib import Path
import bcrypt

# Add server root to path
server_root = Path(__file__).parent
sys.path.insert(0, str(server_root))

from db import get_db, User

def verify_admin_password():
    """Verify the admin password."""
    db = next(get_db())
    try:
        admin = db.query(User).filter(User.username == 'admin').first()
        if not admin:
            print("❌ Admin user not found")
            return
        
        print(f"Admin user found:")
        print(f"  Username: {admin.username}")
        print(f"  User ID: {admin.userId}")
        print(f"  Role: {admin.role}")
        print(f"  Password hash: {admin.hashed_password[:50]}...")
        
        # Test password verification
        password = "password"
        is_valid = bcrypt.checkpw(password.encode('utf-8'), admin.hashed_password.encode('utf-8'))
        print(f"\nPassword 'password' valid: {is_valid}")
        
        if not is_valid:
            print("Resetting admin password...")
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            admin.hashed_password = hashed.decode('utf-8')
            db.commit()
            print("✅ Admin password reset to 'password'")
        
    finally:
        db.close()

if __name__ == "__main__":
    verify_admin_password()
