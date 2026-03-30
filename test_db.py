"""
Database initialization test script
Run this to verify tables are created properly
"""

import sys
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("DATABASE INITIALIZATION TEST")
print("=" * 60)

# Display configuration
print("\n📋 Configuration:")
print(f"  DB_USER: {os.getenv('DB_USER', 'root')}")
print(f"  DB_HOST: {os.getenv('DB_HOST', 'localhost')}")
print(f"  DB_PORT: {os.getenv('DB_PORT', '3306')}")
print(f"  DB_NAME: {os.getenv('DB_NAME', 'nemo')}")

# Test database connection
print("\n🔗 Testing database connection...")
try:
    from database import engine
    with engine.connect() as connection:
        print("  ✓ Successfully connected to MySQL")
except Exception as e:
    print(f"  ✗ Connection failed: {str(e)}")
    print("  Make sure MySQL is running and credentials are correct")
    sys.exit(1)

# Initialize database
print("\n📦 Initializing database and creating tables...")
try:
    from database import init_db, Base
    init_db()
    print(f"  ✓ Initialization complete")
except Exception as e:
    print(f"  ✗ Initialization failed: {str(e)}")
    sys.exit(1)

# Verify tables
print("\n✅ Verifying tables...")
try:
    from database import engine
    from sqlalchemy import inspect
    
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    if not tables:
        print("  ✗ No tables found!")
        sys.exit(1)
    
    print(f"  ✓ Found {len(tables)} table(s):")
    for table in tables:
        columns = [col['name'] for col in inspector.get_columns(table)]
        print(f"    - {table}: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
    
    print("\n✓ Database setup successful!")
    
except Exception as e:
    print(f"  ✗ Verification failed: {str(e)}")
    sys.exit(1)
