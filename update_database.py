#!/usr/bin/env python3
"""
GAM Portfolio Database Update Script
===================================

This script automates the complete database update process when new GAM returns
are added to the Excel spreadsheet.

Process Flow:
1. Impute missing market data using Tingo
2. Migrate Excel data to SQLite database
3. Migrate SQLite data to PostgreSQL (Supabase)

Usage:
    python update_database.py

Prerequisites:
- Updated GAM.xlsx file with new monthly returns
- All required Python packages installed
- Supabase credentials configured in environment

Author: Portfolio Analytics Team
Last Updated: September 2025
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DatabaseUpdater:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.excel_file = self.script_dir / "GAM.xlsx"
        self.sqlite_db = self.script_dir / "portfolio_data.db"
        
    def check_prerequisites(self):
        """Check if all required files and dependencies exist"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check if Excel file exists
        if not self.excel_file.exists():
            raise FileNotFoundError(f"❌ Excel file not found: {self.excel_file}")
        
        # Check if required Python scripts exist
        required_scripts = [
            "impute_data.py",
            "excel_to_database.py"
        ]
        
        for script in required_scripts:
            script_path = self.script_dir / script
            if not script_path.exists():
                raise FileNotFoundError(f"❌ Required script not found: {script_path}")
        
        logger.info("✅ All prerequisites check passed")
        
    def run_script(self, script_name, description):
        """Run a Python script and handle errors"""
        logger.info(f"🚀 Starting: {description}")
        
        try:
            script_path = self.script_dir / script_name
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.script_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"✅ Completed: {description}")
            
            # Log output if there's any
            if result.stdout.strip():
                logger.info(f"Output: {result.stdout.strip()}")
                
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed: {description}")
            logger.error(f"Error code: {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            if e.stdout:
                logger.error(f"Standard output: {e.stdout}")
            return False
        
    def run_postgres_migration(self):
        """Run PostgreSQL migration if script exists"""
        migrate_script = self.script_dir / "migrate_to_postgres.py"
        
        if migrate_script.exists():
            return self.run_script("migrate_to_postgres.py", "Migrating data to PostgreSQL (Supabase)")
        else:
            logger.warning("⚠️ migrate_to_postgres.py not found - skipping PostgreSQL migration")
            logger.info("💡 Data is available in SQLite database for local use")
            return True
    
    def backup_database(self):
        """Create a backup of the current database"""
        if self.sqlite_db.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.script_dir / f"portfolio_data_backup_{timestamp}.db"
            
            try:
                import shutil
                shutil.copy2(self.sqlite_db, backup_path)
                logger.info(f"📦 Database backup created: {backup_path.name}")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Failed to create backup: {e}")
                return False
        return True
    
    def update_database(self):
        """Main method to update the database"""
        logger.info("🎯 Starting GAM Portfolio Database Update")
        logger.info("=" * 50)
        
        try:
            # Step 0: Check prerequisites
            self.check_prerequisites()
            
            # Step 1: Create backup
            self.backup_database()
            
            # Step 2: Impute missing market data
            success = self.run_script(
                "impute_data.py", 
                "Imputing missing market data from Tingo"
            )
            if not success:
                logger.error("❌ Failed to impute market data. Stopping process.")
                return False
            
            # Step 3: Migrate Excel to SQLite
            success = self.run_script(
                "excel_to_database.py",
                "Migrating Excel data to SQLite database"
            )
            if not success:
                logger.error("❌ Failed to migrate Excel data to SQLite. Stopping process.")
                return False
            
            # Step 4: Migrate to PostgreSQL (optional)
            success = self.run_postgres_migration()
            if not success:
                logger.error("❌ Failed to migrate to PostgreSQL. SQLite data is still available.")
                # Don't return False here - SQLite migration was successful
            
            logger.info("=" * 50)
            logger.info("🎉 Database update completed successfully!")
            logger.info("📊 Dashboard will now reflect the new data")
            
            # Display next steps
            self.display_next_steps()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Unexpected error during database update: {e}")
            return False
    
    def display_next_steps(self):
        """Display next steps for the user"""
        logger.info("\n📋 Next Steps:")
        logger.info("1. Restart your Streamlit dashboard to see the new data")
        logger.info("2. Verify the new month's data appears correctly in all charts")
        logger.info("3. Check that attribution and allocation values look reasonable")
        logger.info("4. If using production deployment, ensure PostgreSQL migration was successful")


def main():
    """Main entry point"""
    updater = DatabaseUpdater()
    
    # Display welcome message
    print("\n" + "=" * 60)
    print("🏦 GAM PORTFOLIO DATABASE UPDATE AUTOMATION")
    print("=" * 60)
    print("This script will update your portfolio database with new GAM returns")
    print("Make sure your GAM.xlsx file contains the latest monthly data")
    print("=" * 60 + "\n")
    
    # Confirm before proceeding
    try:
        response = input("Continue with database update? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Database update cancelled by user")
            return
    except KeyboardInterrupt:
        print("\n❌ Database update cancelled by user")
        return
    
    # Run the update
    success = updater.update_database()
    
    if success:
        print("\n🎉 SUCCESS: Database update completed!")
        sys.exit(0)
    else:
        print("\n❌ FAILED: Database update encountered errors")
        print("📋 Check the log file 'database_update.log' for details")
        sys.exit(1)


if __name__ == "__main__":
    main()