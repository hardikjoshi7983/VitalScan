"""MySQL connection helper. The web demo runs without storing patient details."""
import os
from sqlalchemy import create_engine, text

def get_engine():
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "123456")
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = os.getenv("MYSQL_PORT", "3306")
    database = os.getenv("MYSQL_DATABASE", "healthinsight")
    return create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}", pool_pre_ping=True)

def initialize_database():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS assessment_audit (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version VARCHAR(80) NOT NULL,
                note VARCHAR(255) NOT NULL
            )
        """))
    return engine
