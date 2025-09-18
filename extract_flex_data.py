#!/usr/bin/env python3
"""
Script to extract GAM strategy data from daily flex reports and append to portfolio_data.db
Processes raw_RQA_flex_SMA.csv to extract position data, allocations, and performance for GAM strategy
"""

import pandas as pd
import sqlite3
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_flex_csv(file_path):
    """
    Parse the flex CSV file and extract GAM Morningstar data
    """
    logger.info(f"Parsing flex CSV file: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)
    
    # Initialize data containers
    gam_data = {
        'nav_data': [],
        'position_data': [],
        'performance_data': []
    }
    
    report_date = None
    
    # Process each row
    for idx, row in df.iterrows():
        if len(row) < 5:
            continue
            
        row_type = row[0]
        data_type = row[1]
        account_id = row[2]
        model = row[4] if len(row) > 4 else None
        
        # Skip if not GAM Morningstar data
        if model != 'GAMMorningstar':
            continue
            
        # Extract report date from first GAM record
        if report_date is None and len(row) > 7:
            try:
                report_date = row[7]  # ReportDate column
                if isinstance(report_date, str) and len(report_date) == 8:
                    report_date = datetime.strptime(report_date, '%Y%m%d').strftime('%Y-%m-%d')
                logger.info(f"Processing data for report date: {report_date}")
            except:
                pass
        
        # Extract NAV data (CNAV type)
        if row_type == 'DATA' and data_type == 'CNAV':
            try:
                starting_value = float(row[8]) if len(row) > 8 and row[8] else 0
                mtm = float(row[9]) if len(row) > 9 and row[9] else 0
                ending_value = float(row[55]) if len(row) > 55 and row[55] else 0
                twr = float(row[56]) if len(row) > 56 and row[56] else 0
                
                gam_data['nav_data'].append({
                    'account_id': account_id,
                    'report_date': report_date,
                    'starting_value': starting_value,
                    'mtm_pnl': mtm,
                    'ending_value': ending_value,
                    'twr': twr
                })
                logger.info(f"NAV data - Account: {account_id}, Ending Value: ${ending_value:,.2f}, MTM: ${mtm:.2f}")
            except Exception as e:
                logger.warning(f"Error parsing NAV data: {e}")
        
        # Extract position data (POST type)
        elif row_type == 'DATA' and data_type == 'POST':
            try:
                symbol = row[9] if len(row) > 9 else None
                description = row[10] if len(row) > 10 else None
                quantity = float(row[31]) if len(row) > 31 and row[31] else 0
                mark_price = float(row[32]) if len(row) > 32 and row[32] else 0
                position_value = float(row[33]) if len(row) > 33 and row[33] else 0
                percent_of_nav = float(row[37]) if len(row) > 37 and row[37] else 0
                
                # Calculate position value from quantity * mark_price if position_value is 0
                if position_value == 0 and quantity > 0 and mark_price > 0:
                    position_value = quantity * mark_price
                
                if symbol:  # Include all positions including cash (USD)
                    # For cash positions, use special handling
                    if symbol == 'USD':
                        symbol = 'CASH'  # Rename USD to CASH for consistency
                        description = 'Cash Holdings'
                        mark_price = 1.0  # Cash has a mark price of 1.0
                        if position_value == 0:
                            position_value = quantity  # For cash, quantity equals value
                    
                    gam_data['position_data'].append({
                        'account_id': account_id,
                        'report_date': report_date,
                        'symbol': symbol,
                        'description': description,
                        'quantity': quantity,
                        'mark_price': mark_price,
                        'position_value': position_value,
                        'percent_of_nav': percent_of_nav
                    })
                    
                    if symbol == 'CASH':
                        logger.info(f"Cash Position: ${position_value:,.2f} ({percent_of_nav:.2f}%)")
                    else:
                        logger.info(f"Position - {symbol}: {quantity:.4f} shares @ ${mark_price:.2f} = ${position_value:,.2f} ({percent_of_nav:.2f}%)")
            except Exception as e:
                logger.warning(f"Error parsing position data: {e}")
        
        # Extract MTM performance data (MTMP type)
        elif row_type == 'DATA' and data_type == 'MTMP':
            try:
                symbol = row[7] if len(row) > 7 else None
                if symbol and symbol != '':  # Skip total P/L row
                    prev_close_qty = float(row[27]) if len(row) > 27 and row[27] else 0
                    prev_close_price = float(row[28]) if len(row) > 28 and row[28] else 0
                    close_qty = float(row[29]) if len(row) > 29 and row[29] else 0
                    close_price = float(row[30]) if len(row) > 30 and row[30] else 0
                    mtm_pnl = float(row[37]) if len(row) > 37 and row[37] else 0
                    
                    gam_data['performance_data'].append({
                        'account_id': account_id,
                        'report_date': report_date,
                        'symbol': symbol,
                        'prev_close_qty': prev_close_qty,
                        'prev_close_price': prev_close_price,
                        'close_qty': close_qty,
                        'close_price': close_price,
                        'daily_mtm_pnl': mtm_pnl
                    })
            except Exception as e:
                logger.warning(f"Error parsing MTM data: {e}")
    
    return gam_data, report_date

def update_database(gam_data, report_date):
    """
    Update the portfolio_data.db with new GAM flex data
    """
    logger.info("Updating database with new GAM data...")
    
    conn = sqlite3.connect('portfolio_data.db')
    cursor = conn.cursor()
    
    # Create new tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gam_daily_nav (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT,
            report_date DATE,
            starting_value REAL,
            mtm_pnl REAL,
            ending_value REAL,
            twr REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(account_id, report_date)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gam_daily_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT,
            report_date DATE,
            symbol TEXT,
            description TEXT,
            quantity REAL,
            mark_price REAL,
            position_value REAL,
            percent_of_nav REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(account_id, report_date, symbol)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gam_daily_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT,
            report_date DATE,
            symbol TEXT,
            prev_close_qty REAL,
            prev_close_price REAL,
            close_qty REAL,
            close_price REAL,
            daily_mtm_pnl REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(account_id, report_date, symbol)
        )
    ''')
    
    # Insert NAV data
    nav_inserted = 0
    for nav_record in gam_data['nav_data']:
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO gam_daily_nav 
                (account_id, report_date, starting_value, mtm_pnl, ending_value, twr)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                nav_record['account_id'],
                nav_record['report_date'],
                nav_record['starting_value'],
                nav_record['mtm_pnl'],
                nav_record['ending_value'],
                nav_record['twr']
            ))
            nav_inserted += 1
        except Exception as e:
            logger.error(f"Error inserting NAV data: {e}")
    
    # Insert position data
    positions_inserted = 0
    for position in gam_data['position_data']:
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO gam_daily_positions 
                (account_id, report_date, symbol, description, quantity, mark_price, position_value, percent_of_nav)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position['account_id'],
                position['report_date'],
                position['symbol'],
                position['description'],
                position['quantity'],
                position['mark_price'],
                position['position_value'],
                position['percent_of_nav']
            ))
            positions_inserted += 1
        except Exception as e:
            logger.error(f"Error inserting position data: {e}")
    
    # Insert performance data
    performance_inserted = 0
    for perf in gam_data['performance_data']:
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO gam_daily_performance 
                (account_id, report_date, symbol, prev_close_qty, prev_close_price, close_qty, close_price, daily_mtm_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                perf['account_id'],
                perf['report_date'],
                perf['symbol'],
                perf['prev_close_qty'],
                perf['prev_close_price'],
                perf['close_qty'],
                perf['close_price'],
                perf['daily_mtm_pnl']
            ))
            performance_inserted += 1
        except Exception as e:
            logger.error(f"Error inserting performance data: {e}")
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database update complete:")
    logger.info(f"  - NAV records inserted: {nav_inserted}")
    logger.info(f"  - Position records inserted: {positions_inserted}")
    logger.info(f"  - Performance records inserted: {performance_inserted}")
    
    return nav_inserted, positions_inserted, performance_inserted

def calculate_portfolio_metrics(report_date):
    """
    Calculate portfolio-level metrics from position data
    """
    logger.info("Calculating portfolio-level metrics...")
    
    conn = sqlite3.connect('portfolio_data.db')
    
    # Get total portfolio value for the date
    total_nav_query = '''
        SELECT account_id, SUM(ending_value) as total_nav
        FROM gam_daily_nav 
        WHERE report_date = ?
        GROUP BY account_id
    '''
    
    # Get position allocations
    allocation_query = '''
        SELECT 
            account_id,
            symbol,
            SUM(position_value) as total_position_value,
            AVG(percent_of_nav) as avg_percent_nav
        FROM gam_daily_positions 
        WHERE report_date = ?
        GROUP BY account_id, symbol
        ORDER BY total_position_value DESC
    '''
    
    nav_df = pd.read_sql_query(total_nav_query, conn, params=[report_date])
    positions_df = pd.read_sql_query(allocation_query, conn, params=[report_date])
    
    logger.info(f"Portfolio metrics for {report_date}:")
    logger.info(f"Total NAV across accounts: ${nav_df['total_nav'].sum():,.2f}")
    logger.info(f"Top positions:")
    for _, row in positions_df.head(10).iterrows():
        logger.info(f"  {row['symbol']}: ${row['total_position_value']:,.2f} ({row['avg_percent_nav']:.2f}%)")
    
    conn.close()
    
    return nav_df, positions_df

def main():
    """
    Main function to process flex data and update database
    """
    logger.info("Starting GAM flex data extraction...")
    
    # Check if input file exists
    flex_file = 'raw_RQA_flex_SMA.csv'
    if not os.path.exists(flex_file):
        logger.error(f"Input file {flex_file} not found!")
        return
    
    try:
        # Parse the flex CSV
        gam_data, report_date = parse_flex_csv(flex_file)
        
        if not report_date:
            logger.error("Could not determine report date from flex file")
            return
        
        # Update database
        nav_count, pos_count, perf_count = update_database(gam_data, report_date)
        
        # Calculate portfolio metrics
        nav_df, positions_df = calculate_portfolio_metrics(report_date)
        
        logger.info("GAM flex data extraction completed successfully!")
        logger.info(f"Summary for {report_date}:")
        logger.info(f"  - GAM accounts processed: {len(set([n['account_id'] for n in gam_data['nav_data']]))}")
        logger.info(f"  - Total positions: {len(gam_data['position_data'])}")
        logger.info(f"  - Total portfolio value: ${nav_df['total_nav'].sum():,.2f}")
        
    except Exception as e:
        logger.error(f"Error processing flex data: {e}")
        raise

if __name__ == "__main__":
    main()
