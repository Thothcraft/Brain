#!/usr/bin/env python3
"""Check if admin user exists."""

import sys
from pathlib import Path

# Add server root to path
server_root = Path(__file__).parent
sys.path.insert(0, str(server_root))

from db import get_db, User

def check_admin():
    """Check if admin user exists."""
    db = next(get_db())
    try:
        admin = db.query(User).filter(User.username == 'admin').first()
        if admin:
            print(f"✅ Admin user exists:")
            print(f"   Username: {admin.username}")
            print(f"   User ID: {admin.userId}")
            print(f"   Role: {admin.role}")
            print(f"   Plan: {admin.plan}")
            return True
        else:
            print("❌ Admin user does not exist")
            return False
    finally:
        db.close()

if __name__ == "__main__":
    check_admin()
