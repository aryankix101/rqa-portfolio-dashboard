#!/usr/bin/env python3
"""
Monthly GAM data processor - automates the monthly extraction and integration of GAM flex data
Run this script at month-end to update the portfolio database with latest GAM positions and performance
"""

import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta
import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_flex_data import parse_flex_csv, update_database, calculate_portfolio_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_monthly_returns(report_date):
    """
    Calculate monthly returns from the daily NAV data and update monthly_returns table
    """
    logger.info(f"Calculating monthly returns for {report_date}")
    
    conn = sqlite3.connect('portfolio_data.db')
    cursor = conn.cursor()
    
    # Get total portfolio NAV for the current month
    current_month_query = '''
        SELECT SUM(ending_value) as total_nav
        FROM gam_daily_nav 
        WHERE report_date = ?
    '''
    
    # Get previous month-end NAV (approximate)
    prev_month_date = datetime.strptime(report_date, '%Y-%m-%d') - timedelta(days=30)
    prev_month_str = prev_month_date.strftime('%Y-%m-%d')
    
    # Look for the closest previous month-end data
    prev_month_query = '''
        SELECT SUM(ending_value) as total_nav, report_date
        FROM gam_daily_nav 
        WHERE report_date <= ?
        ORDER BY report_date DESC
        LIMIT 1
    '''
    
    current_nav = pd.read_sql_query(current_month_query, conn, params=[report_date])
    prev_nav = pd.read_sql_query(prev_month_query, conn, params=[prev_month_str])
    
    if len(current_nav) > 0 and len(prev_nav) > 0:
        current_value = current_nav.iloc[0]['total_nav']
        prev_value = prev_nav.iloc[0]['total_nav']
        
        if prev_value and prev_value > 0:
            monthly_return = (current_value - prev_value) / prev_value
            
            # Format the date for the monthly_returns table (last day of month)
            month_end_date = datetime.strptime(report_date, '%Y-%m-%d').strftime('%Y-%m-%d')
            
            # Update the monthly_returns table with GAM data
            cursor.execute('''
                UPDATE monthly_returns 
                SET gam_returns_net = ?
                WHERE date = ?
            ''', (monthly_return, month_end_date))
            
            if cursor.rowcount == 0:
                # Insert new record if it doesn't exist
                cursor.execute('''
                    INSERT INTO monthly_returns (date, gam_returns_net)
                    VALUES (?, ?)
                ''', (month_end_date, monthly_return))
            
            conn.commit()
            logger.info(f"Updated monthly return: {monthly_return*100:.2f}% for {month_end_date}")
            return monthly_return
        else:
            logger.warning("Previous NAV is zero, cannot calculate return")
    else:
        logger.warning("Insufficient data to calculate monthly return")
    
    conn.close()
    return 0.0

def update_allocation_data(report_date):
    """
    Update allocation data in the gam_allocations table from position data
    """
    logger.info(f"Updating allocation data for {report_date}")
    
    conn = sqlite3.connect('portfolio_data.db')
    cursor = conn.cursor()
    
    try:
        # Get aggregated position data for allocation calculation
        query = '''
            SELECT symbol, SUM(position_value) as total_value
            FROM gam_daily_positions 
            WHERE report_date = ?
            GROUP BY symbol
        '''
        
        position_data = pd.read_sql_query(query, conn, params=[report_date])
        
        if len(position_data) == 0:
            logger.warning("No position data found for allocation calculation")
            return {}
        
        # Calculate total portfolio value
        total_value = position_data['total_value'].sum()
        
        # Calculate allocations
        allocation_data = {}
        
        # Clear existing allocations for this date
        cursor.execute('DELETE FROM gam_allocations WHERE date = ?', (report_date,))
        
        # Insert new allocation data
        for _, row in position_data.iterrows():
            symbol = row['symbol']
            value = row['total_value']
            percentage = (value / total_value) * 100 if total_value > 0 else 0
            
            # Get asset name (simplified mapping)
            asset_names = {
                'SPLG': 'SPDR Portfolio S&P 500 ETF',
                'IAU': 'iShares Gold Trust',
                'VEA': 'Vanguard FTSE Developed Markets ETF',
                'VNQ': 'Vanguard Real Estate Index Fund',
                'SCHE': 'Schwab Emerging Markets Equity ETF',
                'COMT': 'iShares GSCI Commodity Dynamic Roll Strategy ETF',
                'TLT': 'iShares 20+ Year Treasury Bond ETF',
                'CASH': 'Cash'  # Add cash mapping
            }
            
            asset_name = asset_names.get(symbol, symbol)
            allocation_data[symbol.lower()] = percentage
            
            cursor.execute('''
                INSERT INTO gam_allocations (date, asset_symbol, asset_name, allocation_percentage)
                VALUES (?, ?, ?, ?)
            ''', (report_date, symbol, asset_name, percentage))
            
            logger.info(f"Allocation - {symbol}: {percentage:.1f}%")
        
        conn.commit()
        logger.info(f"Updated allocations for {len(position_data)} assets")
        return allocation_data
        
    except Exception as e:
        logger.error(f"Error updating allocation data: {e}")
        return {}
    finally:
        conn.close()

def generate_monthly_report(report_date):
    """
    Generate a comprehensive monthly report
    """
    logger.info(f"Generating monthly report for {report_date}")
    
    conn = sqlite3.connect('portfolio_data.db')
    
    # Portfolio summary
    summary_query = '''
        SELECT 
            COUNT(*) as active_accounts,
            SUM(ending_value) as total_nav,
            SUM(mtm_pnl) as total_daily_pnl,
            AVG(twr) as avg_twr
        FROM gam_daily_nav 
        WHERE report_date = ?
    '''
    
    summary = pd.read_sql_query(summary_query, conn, params=[report_date])
    
    # Allocation breakdown
    allocation_query = '''
        SELECT 
            symbol,
            SUM(position_value) as total_value,
            SUM(position_value) / (SELECT SUM(position_value) FROM gam_daily_positions WHERE report_date = ?) * 100 as percent_allocation
        FROM gam_daily_positions 
        WHERE report_date = ?
        GROUP BY symbol
        ORDER BY total_value DESC
    '''
    
    allocations = pd.read_sql_query(allocation_query, conn, params=[report_date, report_date])
    
    # Performance by account
    performance_query = '''
        SELECT account_id, ending_value, mtm_pnl, twr
        FROM gam_daily_nav 
        WHERE report_date = ?
        ORDER BY ending_value DESC
    '''
    
    performance = pd.read_sql_query(performance_query, conn, params=[report_date])
    
    conn.close()
    
    # Generate report
    report = f"""
=== GAM STRATEGY MONTHLY REPORT ===
Report Date: {report_date}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO SUMMARY:
- Active Accounts: {summary.iloc[0]['active_accounts']:.0f}
- Total NAV: ${summary.iloc[0]['total_nav']:,.2f}
- Daily P&L: ${summary.iloc[0]['total_daily_pnl']:,.2f}
- Average TWR: {summary.iloc[0]['avg_twr']*100:.3f}%

ASSET ALLOCATION:
"""
    
    for _, row in allocations.iterrows():
        report += f"- {row['symbol']}: {row['percent_allocation']:.1f}% (${row['total_value']:,.0f})\n"
    
    report += f"\nTOP 10 ACCOUNTS BY NAV:\n"
    for _, row in performance.head(10).iterrows():
        report += f"- {row['account_id']}: ${row['ending_value']:,.2f} (P&L: ${row['mtm_pnl']:,.2f})\n"
    
    # Save report to file
    report_filename = f"gam_monthly_report_{report_date.replace('-', '')}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    logger.info(f"Monthly report saved to {report_filename}")
    print(report)
    
    return report

def main():
    """
    Main monthly processing function
    """
    logger.info("Starting monthly GAM data processing...")
    
    # Check if flex file exists
    flex_file = 'raw_RQA_flex_SMA.csv'
    if not os.path.exists(flex_file):
        logger.error(f"Flex file {flex_file} not found!")
        return
    
    try:
        # 1. Extract flex data
        gam_data, report_date = parse_flex_csv(flex_file)
        if not report_date:
            logger.error("Could not determine report date")
            return
        
        # 2. Update database
        nav_count, pos_count, perf_count = update_database(gam_data, report_date)
        
        # 3. Calculate monthly returns
        monthly_return = calculate_monthly_returns(report_date)
        
        # 4. Update allocation data  
        allocation_data = update_allocation_data(report_date)
        
        # 5. Calculate portfolio metrics
        nav_df, positions_df = calculate_portfolio_metrics(report_date)
        
        # 6. Generate monthly report
        report = generate_monthly_report(report_date)
        
        logger.info("Monthly processing completed successfully!")
        logger.info(f"Data updated for {report_date}")
        logger.info(f"Monthly return: {monthly_return*100:.2f}%" if monthly_return else "Monthly return: N/A")
        
    except Exception as e:
        logger.error(f"Error in monthly processing: {e}")
        raise

if __name__ == "__main__":
    main()
