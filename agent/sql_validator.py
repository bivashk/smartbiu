"""
SQL Security & AST Validator using SQLGlot
Guarantees read-only execution, prevents SQL injection, enforces row limits,
and catches syntax errors prior to database execution.
"""

from typing import Tuple, Optional
import sqlglot
from sqlglot import exp

FORBIDDEN_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Command
)

def validate_and_sanitize_sql(query: str, max_rows: int = 100) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Parses and sanitizes SQL query using AST.
    Returns:
        (is_valid: bool, sanitized_sql: Optional[str], error_message: Optional[str])
    """
    clean_query = query.strip().rstrip(";")
    
    # 1. Parse AST
    try:
        parsed = sqlglot.parse_one(clean_query, read="sqlite")
    except Exception as e:
        return False, None, f"SQL Syntax Error: {str(e)}"
        
    # 2. Check for forbidden statements (DML / DDL)
    if any(parsed.find(expr_type) for expr_type in FORBIDDEN_EXPRESSIONS):
        return False, None, "Security Violation: Non-SELECT or mutating SQL statements are strictly prohibited."
        
    # Check that root expression is a Select or Union
    if not isinstance(parsed, (exp.Select, exp.Union)):
        return False, None, "Security Violation: Only SELECT queries are permitted."
        
    # 3. Enforce LIMIT clause if missing to avoid memory denial-of-service
    if not parsed.find(exp.Limit):
        # Only add LIMIT if it's not a single-row aggregation without GROUP BY
        has_group_by = bool(parsed.find(exp.Group))
        selects = parsed.expressions
        has_aggregations = any(s.find(exp.AggFunc) for s in selects)
        
        # If it's a grouped query or non-aggregated query, enforce limit
        if has_group_by or not has_aggregations:
            parsed = parsed.limit(max_rows)
            
    sanitized_sql = parsed.sql(dialect="sqlite")
    return True, sanitized_sql, None

if __name__ == "__main__":
    # Test valid query
    valid, sql, err = validate_and_sanitize_sql("SELECT customer_id, name FROM customers")
    print("Test 1 (Valid query with auto-limit):", valid, sql)
    
    # Test malicious query
    valid, sql, err = validate_and_sanitize_sql("DROP TABLE customers;")
    print("Test 2 (Malicious DROP):", valid, err)
    
    # Test update query
    valid, sql, err = validate_and_sanitize_sql("UPDATE loans SET loan_amount = 0")
    print("Test 3 (Forbidden UPDATE):", valid, err)
