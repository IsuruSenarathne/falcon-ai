"""
Database configuration and initialization
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool
import os

# MySQL Database Configuration
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "nemo")

# Build MySQL connection URL
if DB_PASSWORD:
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DATABASE_URL = f"mysql+pymysql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Test connections before using
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Import models to register them with Base
# This MUST happen after Base is created but before init_db() is called
from models import Conversation, Message  # noqa: F401, E402


def init_db():
    """
    Initialize database by creating all tables.
    This function is safe to call multiple times - it will only create
    tables that don't already exist.
    """
    try:
        Base.metadata.create_all(bind=engine)
        print(f"✓ Database initialized successfully at {DB_HOST}:{DB_PORT}/{DB_NAME}")
        print(f"✓ Tables created: {list(Base.metadata.tables.keys())}")
    except Exception as e:
        print(f"✗ Error initializing database: {str(e)}")
        raise


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
