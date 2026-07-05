"""
Database Utilities and Schema Introspection for Retail Credit Risk BIU
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retail_credit.db")

def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def execute_query(sql: str, params: Optional[tuple] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Executes a SELECT query and returns rows as a list of dictionaries."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()

def get_schema_summary(db_path: Optional[str] = None) -> str:
    """
    Returns a clean, LLM-ready markdown summary of all tables, columns, data types, 
    and primary/foreign keys to power Schema Linking in Text-to-SQL.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r[0] for r in cursor.fetchall()]
    
    schema_lines = ["# Retail Credit Finance Database Schema\n"]
    
    for table in tables:
        schema_lines.append(f"### Table: `{table}`")
        cursor.execute(f"PRAGMA table_info({table});")
        cols = cursor.fetchall()
        col_desc = []
        for col in cols:
            cid, name, col_type, notnull, dflt_value, pk = col
            pk_badge = " [PRIMARY KEY]" if pk else ""
            nn_badge = " NOT NULL" if notnull else ""
            col_desc.append(f"  - `{name}` ({col_type}{pk_badge}{nn_badge})")
        schema_lines.extend(col_desc)
        
        # Sample row
        cursor.execute(f"SELECT * FROM {table} LIMIT 1;")
        sample = cursor.fetchone()
        if sample:
            sample_dict = dict(sample)
            schema_lines.append(f"  *Sample Record:* `{sample_dict}`\n")
            
    conn.close()
    return "\n".join(schema_lines)

if __name__ == "__main__":
    print(get_schema_summary())
