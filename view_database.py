import sqlite3
import pandas as pd

def view_database():
    conn = sqlite3.connect('portfolio_data.db')
    
    # Set pandas display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)

    print("="*100)
    print("DATABASE VISUALIZATION - LAST 20 RECORDS FOR EACH TABLE")
    print("="*100)
    
    print("\n🔸 MONTHLY RETURNS (Last 20 records)")
    print("-" * 80)
    monthly_returns = pd.read_sql_query("""
        SELECT * FROM monthly_returns 
        ORDER BY date DESC 
        LIMIT 20
    """, conn)
    print(monthly_returns.to_string(index=False))
    
    print(f"\nTotal Monthly Return Records: {len(pd.read_sql_query('SELECT * FROM monthly_returns', conn))}")
    
    print("\n🔸 BENCHMARK PERFORMANCE (All records)")
    print("-" * 80)
    benchmark_performance = pd.read_sql_query("SELECT * FROM benchmark_performance", conn)
    print(benchmark_performance.to_string(index=False))
    
    print(f"\nTotal Benchmark Performance Records: {len(benchmark_performance)}")

    print("\n🔸 GAM ALLOCATIONS (Last 20 records)")
    print("-" * 80)
    gam_allocations = pd.read_sql_query("""
        SELECT date, asset_symbol, asset_name, 
               ROUND(allocation_percentage * 100, 2) as allocation_pct
        FROM gam_allocations 
        ORDER BY date DESC, asset_symbol 
        LIMIT 20
    """, conn)
    print(gam_allocations.to_string(index=False))
    
    print(f"\nTotal GAM Allocation Records: {len(pd.read_sql_query('SELECT * FROM gam_allocations', conn))}")

    print("\n🔸 GAM ATTRIBUTION (Last 20 records)")
    print("-" * 80)
    gam_attribution = pd.read_sql_query("""
        SELECT date, asset_symbol, asset_name, 
               ROUND(attribution_value * 10000, 2) as attribution_bps
        FROM gam_attribution 
        ORDER BY date DESC, asset_symbol 
        LIMIT 20
    """, conn)
    print(gam_attribution.to_string(index=False))
    
    print(f"\nTotal GAM Attribution Records: {len(pd.read_sql_query('SELECT * FROM gam_attribution', conn))}")

    # Show latest month allocation breakdown for verification
    print("\n🔸 LATEST MONTH ALLOCATION BREAKDOWN (Should sum to ~100%)")
    print("-" * 80)
    latest_allocations = pd.read_sql_query("""
        SELECT asset_name, 
               ROUND(allocation_percentage * 100, 2) as allocation_pct
        FROM gam_allocations 
        WHERE date = (SELECT MAX(date) FROM gam_allocations)
        ORDER BY allocation_percentage DESC
    """, conn)
    print(latest_allocations.to_string(index=False))
    total_allocation = latest_allocations['allocation_pct'].sum()
    print(f"Total Allocation: {total_allocation:.2f}%")
    
    # Show date ranges
    print("\n🔸 DATE RANGES")
    print("-" * 80)
    date_info = pd.read_sql_query("""
        SELECT 
            'Monthly Returns' as table_name,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(*) as record_count
        FROM monthly_returns
        UNION ALL
        SELECT 
            'GAM Allocations' as table_name,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(DISTINCT date) as record_count
        FROM gam_allocations
        UNION ALL
        SELECT 
            'GAM Attribution' as table_name,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(DISTINCT date) as record_count
        FROM gam_attribution
    """, conn)
    print(date_info.to_string(index=False))
    
    print("\n" + "="*100)
    print("VERIFICATION SUMMARY")
    print("="*100)
    print("✅ Monthly Returns: Raw return data for GAM and benchmarks")
    print("✅ Benchmark Performance: Calculated metrics (YTD, 1-year, 5-year, etc.)")
    print("✅ GAM Allocations: Asset allocation percentages (for stacked bar chart)")
    print("✅ GAM Attribution: Asset contribution values (for attribution chart)")
    print("="*100)
    
    conn.close()

if __name__ == "__main__":
    view_database()
