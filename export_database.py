#!/usr/bin/env python3
"""
Database Export Script for RQA Portfolio Database
Exports all tables to visual/tabular representations in multiple formats
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime
import json

def get_all_tables(db_path):
    """Get list of all tables in the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def get_table_info(db_path, table_name):
    """Get schema information for a table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]
    conn.close()
    
    return {
        'schema': schema,
        'row_count': row_count
    }

def export_table_to_csv(db_path, table_name, output_dir):
    """Export a table to CSV format"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    
    csv_path = os.path.join(output_dir, f"{table_name}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, df

def export_table_to_excel(db_path, output_dir):
    """Export all tables to a single Excel file with multiple sheets"""
    conn = sqlite3.connect(db_path)
    tables = get_all_tables(db_path)
    
    excel_path = os.path.join(output_dir, "portfolio_database_export.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            # Excel sheet names can't be longer than 31 characters
            sheet_name = table[:31] if len(table) > 31 else table
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    conn.close()
    return excel_path

def create_html_report(db_path, output_dir):
    """Create an HTML report with all tables"""
    conn = sqlite3.connect(db_path)
    tables = get_all_tables(db_path)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RQA Portfolio Database Export</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #3b5998; text-align: center; }}
            h2 {{ color: #1e3a8a; border-bottom: 2px solid #3b82f6; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #3b5998; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .table-info {{ background-color: #e3f2fd; padding: 10px; margin-bottom: 10px; border-radius: 5px; }}
            .timestamp {{ text-align: center; color: #666; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>RQA Portfolio Database Export</h1>
        <div class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    """
    
    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        table_info = get_table_info(db_path, table)
        
        html_content += f"""
        <h2>Table: {table}</h2>
        <div class="table-info">
            <strong>Rows:</strong> {table_info['row_count']} | 
            <strong>Columns:</strong> {len(table_info['schema'])}
        </div>
        """
        
        # Add table data
        html_content += df.to_html(classes='data-table', table_id=f'table-{table}', escape=False)
        html_content += "<br><br>"
    
    html_content += """
    </body>
    </html>
    """
    
    html_path = os.path.join(output_dir, "database_report.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    conn.close()
    return html_path

def create_summary_report(db_path, output_dir):
    """Create a summary report with database statistics"""
    conn = sqlite3.connect(db_path)
    tables = get_all_tables(db_path)
    
    summary_data = []
    for table in tables:
        info = get_table_info(db_path, table)
        summary_data.append({
            'Table Name': table,
            'Row Count': info['row_count'],
            'Column Count': len(info['schema']),
            'Columns': ', '.join([col[1] for col in info['schema']])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "database_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    conn.close()
    return summary_path, summary_df

def export_database(db_path="portfolio_data.db", output_dir="database_exports"):
    """Main function to export database in multiple formats"""
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"🚀 Starting database export from: {db_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Get all tables
    tables = get_all_tables(db_path)
    print(f"📊 Found {len(tables)} tables: {', '.join(tables)}")
    
    exported_files = []
    
    # Export individual CSV files
    print("\n📋 Exporting individual CSV files...")
    csv_dir = os.path.join(output_dir, "csv_files")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    for table in tables:
        csv_path, df = export_table_to_csv(db_path, table, csv_dir)
        print(f"  ✅ {table}: {len(df)} rows → {csv_path}")
        exported_files.append(csv_path)
    
    # Export Excel file
    print("\n📈 Exporting Excel file...")
    excel_path = export_table_to_excel(db_path, output_dir)
    print(f"  ✅ Excel export → {excel_path}")
    exported_files.append(excel_path)
    
    # Create HTML report
    print("\n🌐 Creating HTML report...")
    html_path = create_html_report(db_path, output_dir)
    print(f"  ✅ HTML report → {html_path}")
    exported_files.append(html_path)
    
    # Create summary report
    print("\n📊 Creating summary report...")
    summary_path, summary_df = create_summary_report(db_path, output_dir)
    print(f"  ✅ Summary report → {summary_path}")
    exported_files.append(summary_path)
    
    # Print summary
    print(f"\n🎉 Export complete!")
    print(f"📁 All files saved to: {os.path.abspath(output_dir)}")
    print(f"📄 {len(exported_files)} files created")
    
    # Display table summary
    print(f"\n📋 Database Summary:")
    print(summary_df.to_string(index=False))
    
    return exported_files

def view_table_interactive(db_path="portfolio_data.db", table_name=None):
    """Interactive table viewer"""
    conn = sqlite3.connect(db_path)
    tables = get_all_tables(db_path)
    
    if table_name is None:
        print("Available tables:")
        for i, table in enumerate(tables, 1):
            info = get_table_info(db_path, table)
            print(f"  {i}. {table} ({info['row_count']} rows)")
        
        choice = input("\nEnter table number or name: ").strip()
        
        if choice.isdigit():
            table_name = tables[int(choice) - 1]
        else:
            table_name = choice
    
    if table_name in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        print(f"\n📊 Table: {table_name}")
        print(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\n{df.to_string()}")
        
        # Show data types
        print(f"\n📋 Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
    else:
        print(f"❌ Table '{table_name}' not found!")
    
    conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "view":
            table_name = sys.argv[2] if len(sys.argv) > 2 else None
            view_table_interactive(table_name=table_name)
        elif sys.argv[1] == "export":
            export_database()
        else:
            print("Usage:")
            print("  python export_database.py export  # Export all tables")
            print("  python export_database.py view [table_name]  # View table interactively")
    else:
        # Default: export everything
        export_database()