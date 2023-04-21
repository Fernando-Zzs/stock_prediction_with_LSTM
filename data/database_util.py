# -*- coding = utf-8 -*-
import sqlite3

import pandas as pd

db_name = "stock.db"


def save_to_db(df, table_name, with_index):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS '{table_name}'")
    df.to_sql(table_name, conn, if_exists='replace', index=with_index)
    conn.commit()
    conn.close()


def get_from_db(stock_code):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(f"SELECT * from '{stock_code}'", conn)
    conn.close()
    return df
