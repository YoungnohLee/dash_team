import sqlite3
import pandas as pd
import threading
import queue
import time

test_db = "TEST.DB"
train_db = "TRAIN.DB"

# SQLite 데이터베이스 연결 설정
MAX_CONNECTIONS = 5

# SQLite 연결을 관리할 Connection Pool 클래스
class ConnectionPool:
    def __init__(self, database):
        self._connections = queue.Queue(maxsize=MAX_CONNECTIONS)
        self._lock = threading.Lock()
        for _ in range(MAX_CONNECTIONS):
            self._connections.put(sqlite3.connect(database, check_same_thread=False))
    
    def get_connection(self):
        return self._connections.get()

    def release_connection(self, conn):
        self._connections.put(conn)

# 단일 Connection Pool 인스턴스 생성
train_conn_pool = ConnectionPool(train_db)
test_conn_pool = ConnectionPool(test_db)

def csv2db():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    train_conn = train_conn_pool.get_connection()
    test_conn = test_conn_pool.get_connection()

    try:
        train_df.to_sql("train_table", train_conn, if_exists="replace", index=False)
        test_df.to_sql("test_table", test_conn, if_exists="replace", index=False)
        print("csv to db")
    except sqlite3.Error as e:
        print(f"Error inserting CSV into database: {e}")
    finally:
        train_conn_pool.release_connection(train_conn)
        test_conn_pool.release_connection(test_conn)

def shoot_row():
    train_conn = train_conn_pool.get_connection()
    test_conn = test_conn_pool.get_connection()

    train_cur = train_conn.cursor()
    test_cur = test_conn.cursor()

    try:
        # Get the maximum ID from train_table
        train_cur.execute("SELECT MAX(IND) FROM train_table")
        last_index = train_cur.fetchone()[0]

        next_index = last_index + 1
        test_cur.execute(f"SELECT * FROM test_table WHERE IND = {next_index}")
        row = test_cur.fetchone()

        if row:
            col_names = [description[0] for description in test_cur.description]
            placeholders = ', '.join(['?'] * len(col_names))
            train_cur.execute(f"INSERT INTO train_table ({', '.join(col_names)}) VALUES ({placeholders})", row)
            train_conn.commit()
            print(f"Row transferred from TEST.DB to TRAIN.DB successfully.")
    except sqlite3.Error as e:
        print(f"Error transferring row: {e}")
    finally:
        train_conn_pool.release_connection(train_conn)
        test_conn_pool.release_connection(test_conn)

def csv2db_v2():
    train_conn = train_conn_pool.get_connection()
    test_conn = test_conn_pool.get_connection()

    try:
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")

        train_df.to_sql("train_table", train_conn, if_exists="replace", index=False)
        test_df.to_sql("test_table", test_conn, if_exists="replace", index=False)
        print("csv to db")
    except sqlite3.Error as e:
        print(f"Error inserting CSV into database: {e}")
    finally:
        train_conn_pool.release_connection(train_conn)
        test_conn_pool.release_connection(test_conn)

def shoot_row_v2():
    train_conn = train_conn_pool.get_connection()
    test_conn = test_conn_pool.get_connection()

    train_cur = train_conn.cursor()
    test_cur = test_conn.cursor()

    try:
        # Get the maximum ID from train_table
        train_cur.execute("SELECT MAX(IND) FROM train_table")
        last_index = train_cur.fetchone()[0]

        next_index = last_index + 1
        test_cur.execute(f"SELECT * FROM test_table WHERE IND = {next_index}")
        row = test_cur.fetchone()

        if row:
            col_names = [description[0] for description in test_cur.description]
            placeholders = ', '.join(['?'] * len(col_names))
            train_cur.execute(f"INSERT INTO train_table ({', '.join(col_names)}) VALUES ({placeholders})", row)
            train_conn.commit()
            print(f"Row transferred from TEST.DB to TRAIN.DB successfully.")
    except sqlite3.Error as e:
        print(f"Error transferring row: {e}")
    finally:
        train_cur.close()
        test_cur.close()
        train_conn_pool.release_connection(train_conn)
        test_conn_pool.release_connection(test_conn)

def main():
    csv2db()
    while True:
        time.sleep(0.01)
        shoot_row()

if __name__ == "__main__":
    main()
