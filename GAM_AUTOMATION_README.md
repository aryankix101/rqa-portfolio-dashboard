# GAM Strategy Monthly Automation System

## Overview
This system automates the monthly processing of GAM (Global Asset Management) strategy data by extracting information from daily flex reports and updating the portfolio database with comprehensive analytics.

## System Components

### 1. Core Scripts

#### extract_flex_data.py
- **Purpose**: Parse raw flex CSV reports to extract GAM Morningstar strategy data
- **Key Functions**:
  - `parse_flex_csv()`: Processes raw_RQA_flex_SMA.csv files
  - `update_database()`: Updates GAM daily tables with parsed data
  - `calculate_portfolio_metrics()`: Computes position values and allocations
- **Output**: Updates `gam_daily_nav`, `gam_daily_positions`, and `gam_daily_performance` tables

#### monthly_gam_processor.py
- **Purpose**: Monthly automation script for comprehensive GAM data processing
- **Key Functions**:
  - `calculate_monthly_returns()`: Calculates month-over-month portfolio returns
  - `update_allocation_data()`: Computes and stores asset allocation percentages
  - `generate_monthly_report()`: Creates detailed monthly performance reports
- **Output**: Monthly report files and updated database tables

### 2. Database Tables Created

#### gam_daily_nav
- Stores daily NAV (Net Asset Value) data for each account
- Fields: account_id, report_date, ending_value, mtm_pnl, created_at

#### gam_daily_positions
- Records individual position holdings across all accounts
- Fields: account_id, report_date, symbol, description, quantity, mark_price, position_value, percent_of_nav

#### gam_daily_performance
- Tracks performance metrics at account level
- Fields: account_id, report_date, twr, mtm_pnl, ending_value, created_at

#### gam_allocations
- Stores asset allocation percentages by date
- Fields: date, asset_symbol, asset_name, allocation_percentage

## Usage Instructions

### Daily Processing
```bash
python extract_flex_data.py
```
This processes the `raw_RQA_flex_SMA.csv` file and extracts all GAM strategy data.

### Monthly Automation
```bash
python monthly_gam_processor.py
```
This runs the complete monthly processing workflow including:
- Data extraction from flex reports
- Monthly return calculations
- Asset allocation updates
- Performance metrics calculation
- Monthly report generation

## Sample Output

### Data Processed (Latest Run)
- **Accounts Processed**: 19 GAM strategy accounts
- **Total Portfolio NAV**: $1,141,220.21
- **Position Records**: 98 individual positions across 7 ETFs
- **Daily P&L**: $3,242.27

### Asset Allocation (Current)
- SPLG (S&P 500): 33.0%
- IAU (Gold): 22.6%
- VEA (Developed Markets): 18.4%
- VNQ (Real Estate): 11.9%
- SCHE (Emerging Markets): 10.6%
- COMT (Commodities): 3.0%
- TLT (Treasury Bonds): 0.5%

### Top Accounts by NAV
1. U2207948: $507,587.55 (P&L: $1,441.11)
2. U3446271: $117,109.86 (P&L: $332.94)
3. U3943512: $77,163.81 (P&L: $219.36)
4. U4736647: $72,742.97 (P&L: $206.77)
5. U7581256: $61,734.15 (P&L: $175.51)

## Files Generated

### Monthly Reports
- Format: `gam_monthly_report_YYYYMMDD.txt`
- Contains: Portfolio summary, asset allocation, top accounts, performance metrics
- Location: Same directory as scripts

### Log Files
- Comprehensive logging of all processing steps
- Error tracking and data validation results
- Performance metrics and calculation details

## Integration with Existing System

This GAM automation system integrates seamlessly with the existing portfolio dashboard and database:

- **Database**: Uses same `portfolio_data.db` SQLite database
- **Dashboard**: GAM data is available for visualization in `portfolio_dashboard.py`
- **Performance Tracking**: Links with existing benchmark and performance calculation systems

## Data Sources

- **Input**: `raw_RQA_flex_SMA.csv` (Interactive Brokers flex report)
- **Strategy Filter**: Processes only "GAM Morningstar" strategy accounts
- **Position Types**: ETF positions (SPLG, IAU, VEA, VNQ, SCHE, COMT, TLT)

## Automation Schedule

**Recommended Workflow**:
1. **Daily**: Run `extract_flex_data.py` when new flex reports are available
2. **Monthly**: Run `monthly_gam_processor.py` at month-end for comprehensive reporting
3. **Dashboard**: View updated data in real-time via Streamlit dashboard

## Error Handling

- Comprehensive logging for troubleshooting
- Graceful handling of missing data
- Data validation for position values and calculations
- Automatic table creation if missing

## Performance Features

- **Efficient Processing**: Processes 19 accounts and 98 positions in seconds
- **Data Integrity**: Validates position values and allocation calculations
- **Incremental Updates**: Only processes new data, avoiding duplicates
- **Comprehensive Metrics**: Calculates TWR, P&L, allocations, and returns

This automated system eliminates manual data entry and provides real-time insights into GAM strategy performance across all managed accounts.
