"""
Monthly Portfolio Data Automation

Runs monthly data pipeline:
1. GA data via IBKR API  
2. GB data via IBKR API
3. Market data imputation (GA)
4. Market data imputation (GB)
5. Database update
6. PostgreSQL migration
"""

import os
import sys
import subprocess
import logging
import schedule
import time
from datetime import datetime
import signal

class MonthlyAutomation:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.setup_logging()
        self.running = True
        
        self.scripts = [
            "ga_integrated_updater.py",
            "gbe_integrated_updater.py",
            "impute_data.py",
            "gbe_impute_data.py",
            "excel_to_database.py",
            "migrate_to_postgres.py"
        ]
    
    def setup_logging(self):
        log_file = os.path.join(self.script_dir, "monthly_automation.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_script(self, script_name):
        script_path = os.path.join(self.script_dir, script_name)
        self.logger.info(f"Starting {script_name}...")
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=self.script_dir,
                capture_output=True,
                text=True,
                timeout=1800
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ {script_name} completed successfully")
                return True
            else:
                self.logger.error(f"❌ {script_name} failed with code {result.returncode}")
                if result.stderr:
                    self.logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ {script_name} timed out after 30 minutes")
            return False
        except Exception as e:
            self.logger.error(f"❌ {script_name} failed: {e}")
            return False
    
    def run_monthly_workflow(self):
        self.logger.info("🚀 Starting Monthly Portfolio Data Update")
        self.logger.info("=" * 50)
        
        start_time = datetime.now()
        
        for i, script in enumerate(self.scripts, 1):
            self.logger.info(f"Step {i}/{len(self.scripts)}: {script}")
            
            if not self.run_script(script):
                self.logger.error(f"Workflow failed at step {i}")
                return False
        
        duration = datetime.now() - start_time
        self.logger.info("=" * 50)
        self.logger.info(f"🎉 Monthly workflow completed in {duration}")
        return True
    
    def run_scheduled_job(self):
        if datetime.now().day == 1:
            self.logger.info("📅 First of month - running workflow")
            self.run_monthly_workflow()
        else:
            self.logger.info("📅 Not first of month - skipping")
    
    def start_scheduler(self):
        self.logger.info("⏰ Scheduling for 8:15 AM on 1st of every month")
        
        schedule.every().day.at("08:15").do(self.run_scheduled_job)
        
        self.logger.info("✅ Scheduler running. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            self.logger.info("🛑 Scheduler stopped")
    
    def signal_handler(self, signum, frame):
        self.running = False

def main():
    automation = MonthlyAutomation()
    signal.signal(signal.SIGINT, automation.signal_handler)
    signal.signal(signal.SIGTERM, automation.signal_handler)
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--daemon":
            automation.start_scheduler()
        elif arg == "--now":
            automation.run_monthly_workflow()
        elif arg in ["--help", "-h"]:
            print(__doc__)
        else:
            print(f"Unknown option: {arg}. Use --help for usage.")
    else:
        automation.run_monthly_workflow()

if __name__ == "__main__":
    main()