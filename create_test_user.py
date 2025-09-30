import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.db import SessionLocal
from server.models.user import User  # Updated import path
from server.core.security import get_password_hash

def create_test_user():
    db = SessionLocal()
    try:
        # Check if test user already exists
        existing_user = db.query(User).filter(User.username == "testuser").first()
        if existing_user:
            print("Test user already exists!")
            return

        # Create test user
        hashed_password = get_password_hash("testpassword")
        test_user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=hashed_password,
            role="student",
            full_name="Test User"
        )
        
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        print("Test user created successfully!")
        print(f"Username: testuser")
        print(f"Password: testpassword")
        
    except Exception as e:
        db.rollback()
        print(f"Error creating test user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_test_user()
