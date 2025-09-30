from sqlalchemy import create_engine, inspect
from server.db import Base, DATABASE_URL

def check_database():
    try:
        # Create an engine
        engine = create_engine(DATABASE_URL)
        
        # Create an inspector
        inspector = inspect(engine)
        
        # Get table names
        table_names = inspector.get_table_names()
        print("Tables in the database:", table_names)
        
        # Check if 'files' table exists
        if 'files' in table_names:
            print("\nColumns in 'files' table:")
            columns = inspector.get_columns('files')
            for column in columns:
                print(f"- {column['name']} ({column['type']})")
        else:
            print("\n'files' table does not exist in the database.")
            
        # Check if 'users' table exists
        if 'users' in table_names:
            print("\nColumns in 'users' table:")
            columns = inspector.get_columns('users')
            for column in columns:
                print(f"- {column['name']} ({column['type']})")
        else:
            print("\n'users' table does not exist in the database.")
            
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    check_database()
