import pandas as pd
import requests
from datetime import datetime
import openpyxl

# --- Tiingo API Configuration ---
API_KEY = '74d79406a6adbd63dfd68b80230cb624ed177ac4'
headers = {'Content-Type': 'application/json', 'Authorization': f'Token {API_KEY}'}

# Define tickers and portfolio weights
tickers = ['ACWI', 'AGG', 'SPY']
portfolios = {
    '50/50': {'ACWI': 0.5, 'AGG': 0.5},
    '60/40': {'ACWI': 0.6, 'AGG': 0.4},
    '70/30': {'ACWI': 0.7, 'AGG': 0.3},
}

# Generate last calendar day of each month from July 2019 to today
start = datetime(2019, 7, 1)
today = datetime.today()
dates = pd.date_range(start, today, freq='M')  # M = month end (calendar)

# Download price data for all tickers using requests and calculate adjusted monthly returns
def get_monthly_adj_returns(ticker):
    url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices'
    params = {
        'startDate': dates[0].strftime('%Y-%m-%d'),
        'endDate': dates[-1].strftime('%Y-%m-%d'),
        'resampleFreq': 'daily',
        'columns': 'adjClose,date'
    }
    r = requests.get(url, headers=headers, params=params)
    data = r.json()
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.set_index('date').sort_index()
    # Get last calendar day of each month
    monthly = df['adjClose'].resample('M').last()
    # Insert the first value as 0.0 for the first month (so 7/31/19 row is included)
    returns = monthly.pct_change()
    if not returns.empty:
        returns.iloc[0] = 0.0
    return returns

returns_dict = {t: get_monthly_adj_returns(t) for t in tickers}

# Calculate portfolio returns
def get_portfolio_return(weights, returns_dict):
    # Align returns
    df = pd.DataFrame({t: returns_dict[t] for t in weights})
    port_ret = (df * pd.Series(weights)).sum(axis=1)
    return port_ret

portfolio_returns = {name: get_portfolio_return(w, returns_dict) for name, w in portfolios.items()}

# Open the Excel file
wb = openpyxl.load_workbook('GAM_new copy.xlsx')
ws = wb.active

# Map columns D-I to tickers/portfolios directly (order: ACWI, AGG, SPY, 50/50, 60/40, 70/30)
col_map = {
    'ACWI': 4,   # D
    'AGG': 5,    # E
    'SPY': 6,    # F
    '50/50': 7,  # G
    '60/40': 8,  # H
    '70/30': 9   # I
}

# Only update rows with the specified dates and columns D-I, round to two decimals
import numpy as np
row_map = {}
for row in range(2, ws.max_row+1):
    date_val = ws.cell(row=row, column=1).value
    if isinstance(date_val, datetime):
        date_str = date_val.strftime('%-m/%-d/%y')
    elif isinstance(date_val, str):
        try:
            dt = datetime.strptime(date_val, '%m/%d/%y')
            date_str = dt.strftime('%-m/%-d/%y')
        except Exception:
            try:
                dt = datetime.strptime(date_val, '%-m/%-d/%y')
                date_str = dt.strftime('%-m/%-d/%y')
            except Exception:
                continue
    else:
        continue
    row_map[date_str] = row

for date in returns_dict['ACWI'].index:
    date_str = date.strftime('%-m/%-d/%y')
    row = row_map.get(date_str)
    if not row:
        continue  # Only update existing rows
    
    # Update ticker returns
    for t in tickers:
        val = returns_dict[t].get(date, None)
        if val is not None and not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            ws.cell(row=row, column=col_map[t]).value = round(val, 4)
    
    # Update portfolio returns
    for p in portfolios:
        val = portfolio_returns[p].get(date, None)
        if val is not None and not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            ws.cell(row=row, column=col_map[p]).value = round(val, 4)

wb.save('GAM_new copy.xlsx')
print("Excel file updated successfully!")