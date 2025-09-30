from sqlalchemy import create_engine, inspect, text
from server.db import DATABASE_URL

def check_file_table():
    try:
        # Create an engine
        engine = create_engine(DATABASE_URL)
        
        # Create a connection
        with engine.connect() as conn:
            # Get table columns
            inspector = inspect(engine)
            columns = inspector.get_columns('file')
            
            print("Columns in 'file' table:")
            for column in columns:
                print(f"- {column['name']} ({column['type']})")
            
            # Count records
            result = conn.execute(text("SELECT COUNT(*) FROM file"))
            count = result.scalar()
            print(f"\nTotal records in 'file' table: {count}")
            
    except Exception as e:
        print(f"Error checking file table: {e}")

if __name__ == "__main__":
    check_file_table()
