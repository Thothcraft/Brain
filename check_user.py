import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection string
DATABASE_URL = "postgresql://postgres.otjjjagmwzswbinwxmfw:thothpassword@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
print(f"Using database URL: {DATABASE_URL}")

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def check_user(username):
    try:
        # Create a new session
        db = SessionLocal()
        
        # Execute raw SQL query using text()
        query = text("""
            SELECT user_id, username, phone_number, role
            FROM user_account 
            WHERE username = :username
        """)
        result = db.execute(query, {"username": username})
        
        user = result.fetchone()
        
        if user:
            print(f"User found: {user}")
            return True
        else:
            print(f"User '{username}' not found")
            return False
            
    except Exception as e:
        print(f"Error checking user: {str(e)}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    username = "testuser_1759194558"
    check_user(username)
