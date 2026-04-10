import pymysql
from app.config.database import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME


def get_pymysql_conn() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=DB_HOST,
        port=int(DB_PORT),
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        autocommit=True,
    )
