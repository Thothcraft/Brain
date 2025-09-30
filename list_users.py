import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.db import SessionLocal, engine
from server.models.user_models import User

def list_users():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        print("\nUsers in the database:")
        print("-" * 50)
        for user in users:
            print(f"ID: {user.userId}")
            print(f"Username: {user.username}")
            print(f"Email: {user.email}")
            print(f"Role: {user.role}")
            print(f"Created At: {user.created_at}")
            print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    list_users()
