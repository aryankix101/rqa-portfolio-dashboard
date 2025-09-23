import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
import openpyxl

# Database setup
def create_database():
    conn = sqlite3.connect('portfolio_data.db')
    cursor = conn.cursor()
    
    # Create monthly_returns table - matches your Monthly Returns chart
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS monthly_returns (
        date TEXT PRIMARY KEY,
        gam_returns_gross REAL,
        gam_returns_net REAL,
        acwi REAL,
        agg REAL,
        spy REAL,
        portfolio_50_50 REAL,
        portfolio_60_40 REAL,
        portfolio_70_30 REAL
    )
    ''')
    
    # Create gam_allocations table - for the Trailing 12 Month Allocations chart
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS gam_allocations (
        date TEXT,
        asset_symbol TEXT,
        asset_name TEXT,
        allocation_percentage REAL,
        PRIMARY KEY (date, asset_symbol)
    )
    ''')
    
    # Create gam_attribution table - for the Portfolio Attribution chart
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS gam_attribution (
        date TEXT,
        asset_symbol TEXT,
        asset_name TEXT,
        attribution_value REAL,
        PRIMARY KEY (date, asset_symbol)
    )
    ''')
    
    # Create benchmark_performance table - for the Benchmark Performance table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS benchmark_performance (
        portfolio TEXT PRIMARY KEY,
        ytd REAL,
        one_year REAL,
        five_year REAL,
        since_inception REAL,
        standard_deviation REAL,
        sharpe_ratio REAL,
        beta_to_sp500 REAL
    )
    ''')
    
    conn.commit()
    return conn

# Read Excel data
def read_excel_data():
    """
    Read GAM.xlsx and extract all data for the 4 charts/tables needed:
    1. Monthly Returns (columns 1-9)
    2. GAM Allocations (columns 10-17) 
    3. GAM Attribution (columns 18-25)
    4. Benchmark data for performance comparison
    """
    wb = openpyxl.load_workbook('GAM.xlsx')
    ws = wb.active
    
    # Asset mapping for display names in charts
    asset_mapping = {
        'PDBC': 'Commodities',
        'VWO': 'EM_Stocks', 
        'IAU': 'Gold',
        'VEA': 'Intl_Dev_Stocks',
        'SPY': 'SP500',
        'SPTL': 'US_LT_Treas',
        'VNQ': 'US_REITs',
        'USFR': 'Cash'
    }
    
    monthly_returns_data = []
    gam_allocations_data = []
    gam_attribution_data = []
    
    # Start from row 3 (row 2 has headers, row 1 has section labels)
    for row in range(3, ws.max_row + 1):
        date_val = ws.cell(row=row, column=1).value
        if date_val is None:
            continue
            
        # Parse date
        if isinstance(date_val, datetime):
            date_str = date_val.strftime('%Y-%m-%d')
        elif isinstance(date_val, str):
            try:
                dt = datetime.strptime(date_val, '%m/%d/%y')
                date_str = dt.strftime('%Y-%m-%d')
            except Exception:
                continue
        else:
            continue
        
        # Convert numeric strings to floats, handle None values
        def safe_float_convert(value):
            if value is None or value == '' or (isinstance(value, str) and value.strip() == ''):
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        # Extract monthly returns data (columns 2-9)
        gam_returns_gross = safe_float_convert(ws.cell(row=row, column=2).value)
        # Calculate GAM Returns (Net) by subtracting the management fee (0.75% annually = 0.0625% monthly)
        monthly_fee = 0.0075 / 12  # 0.75% annual fee converted to monthly
        gam_returns_net = gam_returns_gross - monthly_fee if gam_returns_gross is not None else None
        
        # Convert percentage values to decimal (divide by 100) for consistency
        acwi = safe_float_convert(ws.cell(row=row, column=4).value)
        
        agg = safe_float_convert(ws.cell(row=row, column=5).value)
        
        spy = safe_float_convert(ws.cell(row=row, column=6).value)
        
        port_50_50 = safe_float_convert(ws.cell(row=row, column=7).value)
        
        port_60_40 = safe_float_convert(ws.cell(row=row, column=8).value)
        
        port_70_30 = safe_float_convert(ws.cell(row=row, column=9).value)
        
        monthly_returns_data.append({
            'date': date_str,
            'gam_returns_gross': gam_returns_gross,
            'gam_returns_net': gam_returns_net,
            'acwi': acwi,
            'agg': agg,
            'spy': spy,
            'portfolio_50_50': port_50_50,
            'portfolio_60_40': port_60_40,
            'portfolio_70_30': port_70_30
        })
        
        # Extract GAM allocation data (columns 10-17)
        allocation_columns = [10, 11, 12, 13, 14, 15, 16, 17]  # PDBC, VWO, IAU, VEA, SPY, SPTL, VNQ, USFR
        for col in allocation_columns:
            asset_symbol = ws.cell(row=2, column=col).value  # Get symbol from header row
            allocation_value = safe_float_convert(ws.cell(row=row, column=col).value)
            
            if asset_symbol and allocation_value is not None:
                asset_name = asset_mapping.get(asset_symbol, asset_symbol)
                gam_allocations_data.append({
                    'date': date_str,
                    'asset_symbol': asset_symbol,
                    'asset_name': asset_name,
                    'allocation_percentage': allocation_value
                })
        
        # Extract GAM attribution data (columns 18-25)
        attribution_columns = [18, 19, 20, 21, 22, 23, 24, 25]  # Same assets, attribution values
        for col in attribution_columns:
            asset_symbol = ws.cell(row=2, column=col).value  # Get symbol from header row
            attribution_value = safe_float_convert(ws.cell(row=row, column=col).value)
            
            if asset_symbol and attribution_value is not None:
                asset_name = asset_mapping.get(asset_symbol, asset_symbol)
                gam_attribution_data.append({
                    'date': date_str,
                    'asset_symbol': asset_symbol,
                    'asset_name': asset_name,
                    'attribution_value': attribution_value
                })
    
    return (pd.DataFrame(monthly_returns_data), 
            pd.DataFrame(gam_allocations_data), 
            pd.DataFrame(gam_attribution_data))

# Calculate annualized return
def annualized_return(returns, periods):
    """Calculate annualized return over specified periods using compound returns"""
    if len(returns) < periods:
        return None
    # Get the returns for the specified period
    period_returns = returns.tail(periods)
    # Calculate cumulative return by compounding
    cumulative_return = (1 + period_returns).prod()
    # Annualize the cumulative return
    years = periods / 12
    return cumulative_return ** (1/years) - 1

# Calculate standard deviation (annualized)
def annualized_std(returns):
    return returns.std() * np.sqrt(12)

# Calculate Sharpe ratio using geometric mean compounded and annualized
def sharpe_ratio(returns):
    if len(returns) == 0:
        return None
    
    # Calculate annualized return using geometric mean
    cumulative_return = (1 + returns).prod()
    years = len(returns) / 12
    annualized_return = cumulative_return ** (1/years) - 1 if years > 0 else 0
    
    # Calculate annualized volatility
    annualized_volatility = returns.std() * np.sqrt(12)
    
    # Sharpe ratio = annualized return / annualized volatility
    if annualized_volatility == 0:
        return None
    
    return annualized_return / annualized_volatility

# Calculate beta to S&P 500
def calculate_beta(portfolio_returns, spy_returns):
    if len(portfolio_returns) == 0 or len(spy_returns) == 0:
        return None
    covariance = np.cov(portfolio_returns, spy_returns)[0][1]
    spy_variance = np.var(spy_returns)
    if spy_variance == 0:
        return None
    return covariance / spy_variance

# Calculate YTD return
def ytd_return(returns, dates):
    current_year = datetime.now().year
    ytd_data = returns[dates.dt.year == current_year]
    if len(ytd_data) == 0:
        return None
    return (1 + ytd_data).prod() - 1

# Store data in database
def store_all_data(conn, monthly_returns_df, gam_allocations_df, gam_attribution_df):
    """Store all extracted data into the respective database tables"""
    
    # Store monthly returns
    monthly_returns_df.to_sql('monthly_returns', conn, if_exists='replace', index=False)
    print(f"Stored {len(monthly_returns_df)} monthly return records")
    
    # Store GAM allocations
    gam_allocations_df.to_sql('gam_allocations', conn, if_exists='replace', index=False)
    print(f"Stored {len(gam_allocations_df)} allocation records")
    
    # Store GAM attribution
    gam_attribution_df.to_sql('gam_attribution', conn, if_exists='replace', index=False)
    print(f"Stored {len(gam_attribution_df)} attribution records")

def calculate_benchmark_performance(conn, monthly_returns_df):
    """
    Calculate the Benchmark Performance table data.
    This creates the table with GAM, 60/40, and 70/30 performance metrics.
    """
    
    # Convert date column to datetime for calculations
    df = monthly_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').dropna()
    
    # Define portfolios to analyze (matching your Benchmark Performance table)
    portfolios = {
        'GAM': 'gam_returns_net',
        '60/40': 'portfolio_60_40', 
        '70/30': 'portfolio_70_30'
    }
    
    performance_data = []
    
    for portfolio_name, column_name in portfolios.items():
        returns = df[column_name].dropna()
        if len(returns) == 0:
            continue
        
        dates = df.loc[returns.index, 'date']
        spy_returns = df.loc[returns.index, 'spy'].dropna()
        
        # Calculate all metrics matching your Benchmark Performance table
        ytd = ytd_return(returns, dates)
        one_year = annualized_return(returns, 12)
        five_year = annualized_return(returns, 60)
        since_inception = annualized_return(returns, len(returns))
        standard_deviation = annualized_std(returns)
        sharpe_ratio_val = sharpe_ratio(returns)
        beta = calculate_beta(returns, spy_returns) if column_name != 'spy' else 1.0
        
        performance_data.append({
            'portfolio': portfolio_name,
            'ytd': ytd,
            'one_year': one_year,
            'five_year': five_year,
            'since_inception': since_inception,
            'standard_deviation': standard_deviation,
            'sharpe_ratio': sharpe_ratio_val,
            'beta_to_sp500': beta
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_sql('benchmark_performance', conn, if_exists='replace', index=False)
    print(f"Calculated benchmark performance for {len(performance_data)} portfolios")

def get_trailing_12m_allocations_chart_data(conn, months=12):
    """
    Query function to get GAM allocation data for creating the 
    'Trailing 12 Month Allocations' stacked bar chart.
    """
    query = '''
    SELECT date, asset_name, allocation_percentage
    FROM gam_allocations 
    ORDER BY date DESC, asset_name
    LIMIT ?
    '''
    
    # Get data for charting
    cursor = conn.cursor()
    cursor.execute(query, (months * 8,))  # 8 assets per month
    results = cursor.fetchall()
    
    # Convert to format suitable for stacked bar chart
    chart_data = {}
    for date, asset_name, percentage in results:
        if date not in chart_data:
            chart_data[date] = {}
        chart_data[date][asset_name] = percentage
    
    return chart_data

def get_attribution_chart_data(conn, months=12):
    """
    Query function to get GAM attribution data for creating the 
    'Trailing 12 Month Portfolio Attribution' chart.
    """
    query = '''
    SELECT date, asset_name, attribution_value
    FROM gam_attribution 
    ORDER BY date DESC, asset_name
    LIMIT ?
    '''
    
    cursor = conn.cursor()
    cursor.execute(query, (months * 8,))  # 8 assets per month
    results = cursor.fetchall()
    
    # Convert to format suitable for attribution chart
    chart_data = {}
    for date, asset_name, attribution_value in results:
        if date not in chart_data:
            chart_data[date] = {}
        chart_data[date][asset_name] = attribution_value
    
    return chart_data

def main():
    print("Creating database...")
    conn = create_database()
    
    print("Reading Excel data...")
    monthly_returns_df, gam_allocations_df, gam_attribution_df = read_excel_data()
    
    print("Storing all data...")
    store_all_data(conn, monthly_returns_df, gam_allocations_df, gam_attribution_df)
    
    print("Calculating benchmark performance metrics...")
    calculate_benchmark_performance(conn, monthly_returns_df)
    
    print("\nDatabase creation complete!")
    
    # Display summary statistics
    cursor = conn.cursor()
    
    # Show monthly returns summary
    cursor.execute("SELECT COUNT(*) FROM monthly_returns")
    returns_count = cursor.fetchone()[0]
    print(f"Monthly returns records: {returns_count}")
    
    # Show allocations summary  
    cursor.execute("SELECT COUNT(*) FROM gam_allocations")
    allocations_count = cursor.fetchone()[0]
    print(f"GAM allocation records: {allocations_count}")
    
    # Show attribution summary
    cursor.execute("SELECT COUNT(*) FROM gam_attribution")
    attribution_count = cursor.fetchone()[0]
    print(f"GAM attribution records: {attribution_count}")
    
    # Show benchmark performance results
    cursor.execute("SELECT * FROM benchmark_performance")
    performance_results = cursor.fetchall()
    print("\nBenchmark Performance Results:")
    print("Portfolio | YTD | 1-Year | 5-Year | Since Inception | Std Dev | Sharpe | Beta")
    print("-" * 80)
    for result in performance_results:
        portfolio = result[0]
        ytd = f"{result[1]:.1%}" if result[1] is not None else "N/A"
        one_year = f"{result[2]:.1%}" if result[2] is not None else "N/A"
        five_year = f"{result[3]:.1%}" if result[3] is not None else "N/A"
        since_inception = f"{result[4]:.1%}" if result[4] is not None else "N/A"
        std_dev = f"{result[5]:.1%}" if result[5] is not None else "N/A"
        sharpe = f"{result[6]:.2f}" if result[6] is not None else "N/A"
        beta = f"{result[7]:.2f}" if result[7] is not None else "N/A"
        
        print(f"{portfolio:9} | {ytd:7} | {one_year:6} | {five_year:6} | {since_inception:15} | {std_dev:7} | {sharpe:6} | {beta:4}")
    
    print("\n" + "="*80)
    print("DATA EXTRACTION SUMMARY:")
    print("="*80)
    print("✅ Monthly Returns -> for 'Monthly Returns' table")
    print("✅ GAM Allocations -> for 'Trailing 12 Month Allocations' chart") 
    print("✅ GAM Attribution -> for 'Portfolio Attribution' chart")
    print("✅ Benchmark Performance -> for 'Benchmark Performance' table")
    print("="*80)
    
    conn.close()

if __name__ == "__main__":
    main()