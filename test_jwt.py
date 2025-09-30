import os
from dotenv import load_dotenv
from jose import jwt
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Get the secret key from environment
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")

print(f"SECRET_KEY: {'*' * 10}{SECRET_KEY[-5:] if SECRET_KEY else 'None'}")
print(f"ALGORITHM: {ALGORITHM}")

# Test JWT token creation
try:
    # Create a test token
    to_encode = {"sub": "1108", "username": "testuser_1759194558"}
    expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    
    # Encode the token
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    print(f"\nToken created successfully!")
    print(f"Token: {token}")
    
    # Decode the token to verify
    decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    print("\nDecoded token:")
    for key, value in decoded.items():
        print(f"{key}: {value}")
    
except Exception as e:
    print(f"\nError creating/verifying token: {str(e)}")
