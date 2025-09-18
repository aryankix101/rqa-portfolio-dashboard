import sqlite3
import pandas as pd

def view_full_database():
    conn = sqlite3.connect('portfolio_data.db')
    
    # Set pandas display options to show all data
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)

    print("="*120)
    print("COMPLETE DATABASE CONTENTS - ALL RECORDS")
    print("="*120)
    
    # Get all table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\nFound {len(tables)} tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Show each table completely
    for table_name in [t[0] for t in tables]:
        print(f"\n{'='*60}")
        print(f"TABLE: {table_name.upper()}")
        print(f"{'='*60}")
        
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Get all data
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        print(f"\nTotal Records: {len(df)}")
        print("\nAll Data:")
        print(df.to_string(index=True))
        print("\n")
    
    print("="*120)
    print("DATABASE SUMMARY")
    print("="*120)
    
    # Summary statistics
    for table_name in [t[0] for t in tables]:
        count = len(pd.read_sql_query(f"SELECT * FROM {table_name}", conn))
        print(f"{table_name}: {count} records")
    
    # Show date ranges for time-series tables
    print(f"\nDate Ranges:")
    for table_name in ['monthly_returns', 'gam_allocations', 'gam_attribution']:
        try:
            date_range = pd.read_sql_query(f"""
                SELECT MIN(date) as earliest, MAX(date) as latest, COUNT(DISTINCT date) as unique_dates
                FROM {table_name}
            """, conn)
            print(f"{table_name}: {date_range.iloc[0]['earliest']} to {date_range.iloc[0]['latest']} ({date_range.iloc[0]['unique_dates']} unique dates)")
        except:
            print(f"{table_name}: No date column or table doesn't exist")
    
    conn.close()

if __name__ == "__main__":
    view_full_database()
