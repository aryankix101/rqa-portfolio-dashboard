"""
Migration script to transfer data from SQLite to PostgreSQL (Supabase)
This script will migrate all tables from your local portfolio_data.db to Supabase
"""

import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
import os
import sys

def get_postgres_connection():
    """Get PostgreSQL connection string from secrets"""
    try:
        import toml
        if os.path.exists('secrets.toml'):
            secrets = toml.load('secrets.toml')
            db_url = secrets['database']['db_url']
            print(f"✅ Successfully loaded connection from secrets.toml")
            return db_url
        else:
            print("❌ secrets.toml not found!")
            print("Please make sure secrets.toml exists with your db_url")
            return None
    except ImportError:
        print("❌ toml package not installed. Run: pip install toml")
        return None
    except Exception as e:
        print(f"❌ Error reading secrets: {e}")
        return None

def check_sqlite_database():
    """Check if SQLite database exists and show table info"""
    if not os.path.exists('portfolio_data.db'):
        print("❌ portfolio_data.db not found!")
        print("Please make sure your SQLite database exists in the current directory")
        return False
    
    try:
        conn = sqlite3.connect('portfolio_data.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        
        print(f"✅ Found SQLite database with {len(tables)} tables:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"   - {table[0]}: {count} rows")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error reading SQLite database: {e}")
        return False

def test_postgres_connection(db_url):
    """Test PostgreSQL connection"""
    try:
        print("🔄 Testing PostgreSQL connection...")
        engine = create_engine(db_url, connect_args={
            "connect_timeout": 10,
            "sslmode": "require"
        })
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ PostgreSQL connection successful!")
            print(f"   Database version: {version[:50]}...")
            
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("\n🛠️  Troubleshooting steps:")
        print("   1. Check if your Supabase project is paused - go to supabase.com and resume it")
        print("   2. Verify the connection string is correct in secrets.toml")
        print("   3. Check your internet connection")
        print("   4. Make sure your IP is allowlisted in Supabase Network settings")
        print("   5. Verify SSL mode and port (5432) are correct")
        return False

def migrate_table(table_name, sqlite_conn, postgres_engine):
    """Migrate a single table from SQLite to PostgreSQL"""
    try:
        print(f"📊 Migrating table: {table_name}")
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)
        print(f"   - Read {len(df)} rows from SQLite")
        
        if df.empty:
            print(f"   - Table {table_name} is empty, creating empty table...")
        
        # Convert date columns to proper format if they exist
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                print(f"   - Converted {date_col} to datetime")
            except:
                pass
        
        df.to_sql(
            table_name, 
            postgres_engine, 
            if_exists='replace', 
            index=False, 
            method='multi',
            chunksize=1000
        )
        print(f"   ✅ Successfully migrated {len(df)} rows to PostgreSQL")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error migrating {table_name}: {e}")
        return False

def verify_migration(postgres_engine):
    """Verify the migration was successful"""
    try:
        print("\n🔍 Verifying migration...")
        
        with postgres_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result]
            
            print(f"✅ Found {len(tables)} tables in PostgreSQL:")
            for table_name in tables:
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = count_result.fetchone()[0]
                print(f"   ✅ {table_name}: {count} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying migration: {e}")
        return False

def main():
    """Main migration function"""
    print("🚀 SQLite to PostgreSQL (Supabase) Migration")
    print("=" * 55)
    
    print("Step 1: Checking SQLite database...")
    if not check_sqlite_database():
        sys.exit(1)
    
    print("\nStep 2: Getting PostgreSQL connection...")
    postgres_url = get_postgres_connection()
    if not postgres_url:
        sys.exit(1)
    
    print("\nStep 3: Testing PostgreSQL connection...")
    if not test_postgres_connection(postgres_url):
        print("\n💡 Common solutions:")
        print("   - Ensure your Supabase project is active (not paused)")
        print("   - Check that the connection string in secrets.toml is correct")
        print("   - Verify your network allows connections to Supabase")
        sys.exit(1)
    
    print("\nStep 4: Starting data migration...")
    
    try:
        sqlite_conn = sqlite3.connect('portfolio_data.db')
        postgres_engine = create_engine(postgres_url)
        
        # Get list of tables from SQLite (excluding system tables)
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n📦 Migrating {len(tables)} tables...")
        
        successful_migrations = 0
        for table_name in tables:
            if migrate_table(table_name, sqlite_conn, postgres_engine):
                successful_migrations += 1
        
        sqlite_conn.close()
        postgres_engine.dispose()
        
        print("\nStep 5: Verifying migration...")
        postgres_engine = create_engine(postgres_url)
        verify_migration(postgres_engine)
        postgres_engine.dispose()
        
        print(f"\n🎉 Migration Summary:")
        print(f"   ✅ Successfully migrated: {successful_migrations}/{len(tables)} tables")
        
        if successful_migrations == len(tables):
            print("\n🌟 Migration completed successfully!")
            print("   Your data is now in Supabase PostgreSQL")
            print("   You can deploy your Streamlit app to Community Cloud!")
        else:
            print(f"\n⚠️  {len(tables) - successful_migrations} tables failed to migrate")
            print("   Please check the error messages above")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()