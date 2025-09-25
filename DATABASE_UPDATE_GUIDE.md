# GAM Portfolio Database Update Guide

## Overview
This guide explains how to update the portfolio database when new GAM monthly returns are available.

## Quick Start (New Person Taking Over)

### 1. When New GAM Returns Are Available
1. **Update Excel File**: Add the new monthly returns to `GAM.xlsx`
2. **Run Update Script**: Execute the automated update process
3. **Verify Results**: Check the dashboard for new data

### 2. Simple Command
```bash
python update_database.py
```

That's it! The script handles everything automatically.

## What the Script Does

The `update_database.py` script automates this complete workflow:

```
📊 GAM.xlsx (Updated)
    ↓
🔄 impute_data.py (Get market data from Tingo)
    ↓  
💾 excel_to_database.py (Convert to SQLite)
    ↓
☁️ migrate_to_postgres.py (Upload to Supabase/PostgreSQL)
    ↓
🎯 Portfolio Dashboard (Updated with new data)
```

## Detailed Process

### Step 1: Prepare Excel File
- Open `GAM.xlsx`
- Add new monthly returns in the appropriate columns
- Save the file

### Step 2: Run Update Script
```bash
# Navigate to the project directory
cd /path/to/RQA

# Run the update script
python update_database.py
```

### Step 3: Verify Update
- Restart your Streamlit dashboard
- Check that the new month appears in all charts
- Verify attribution and allocation values look correct

## Script Features

✅ **Automated Error Handling**: Stops if any step fails  
✅ **Comprehensive Logging**: Creates `database_update.log`  
✅ **Database Backup**: Automatically backs up before changes  
✅ **Prerequisite Checking**: Verifies all files exist  
✅ **Clear Progress**: Shows each step as it completes  

## Manual Process (If Needed)

If you need to run steps individually:

```bash
# Step 1: Impute market data
python impute_data.py

# Step 2: Convert Excel to SQLite
python excel_to_database.py

# Step 3: Upload to PostgreSQL (if using production)
python migrate_to_postgres.py
```

## Troubleshooting

### Common Issues

1. **Excel file not found**
   - Make sure `GAM.xlsx` exists in the project directory
   - Check the file name is exactly "GAM.xlsx"

2. **Script fails during imputation**
   - Check internet connection (needed for Tingo data)
   - Verify Excel file has the correct format

3. **PostgreSQL migration fails**
   - Check Supabase credentials in environment variables
   - Local SQLite database will still be updated

### Log Files
- Check `database_update.log` for detailed error messages
- Backup files are created with timestamp: `portfolio_data_backup_YYYYMMDD_HHMMSS.db`

## File Structure
```
RQA/
├── update_database.py          # 🎯 Main automation script
├── GAM.xlsx                    # 📊 Input data file
├── impute_data.py             # 🔄 Market data fetcher
├── excel_to_database.py       # 💾 Excel to SQLite converter
├── migrate_to_postgres.py     # ☁️ PostgreSQL uploader
├── portfolio_dashboard.py     # 📈 Streamlit dashboard
└── portfolio_data.db          # 💾 SQLite database
```

## Environment Setup

Make sure you have these Python packages installed:
```bash
pip install pandas openpyxl yfinance streamlit plotly supabase
```

## Production Deployment

For production use:
1. Set up Supabase credentials as environment variables
2. Run the update script monthly when new GAM data is available
3. The dashboard will automatically reflect the new data

## Support

If you encounter issues:
1. Check the log file: `database_update.log`
2. Verify all prerequisite files exist
3. Test individual scripts if needed
4. Ensure internet connection for market data fetching

---

**Last Updated**: September 2025  
**Script Version**: 1.0