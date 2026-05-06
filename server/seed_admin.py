#!/usr/bin/env python3
"""Seed script to create the default admin user (admin/password).

Run this after database migrations to ensure an admin account exists.
"""

import os
import sys
from pathlib import Path

# Add server root to path so imports work
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


def main():
    """Create admin user if it doesn't exist."""
    print("[seed_admin] Starting admin seed...")
    db = next(get_db())
    try:
        existing = db.query(User).filter(User.username == "admin").first()
        if existing:
            print("[seed_admin] Admin user already exists")
            return

        admin_user = User(
            username="admin",
            hashed_password=get_password_hash("password"),
            role=1,  # admin
            plan="organization",  # give admin org plan for testing
            org_name="Thothcraft Admin",
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        print(f'[seed_admin] Created admin user: userId={admin_user.userId}, username=admin, password=password')
    finally:
        db.close()


if __name__ == "__main__":
    main()
