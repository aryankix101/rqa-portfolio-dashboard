import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import os
import base64
from io import BytesIO

# PostgreSQL imports with error handling
try:
    from sqlalchemy import create_engine, text
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False





# Configure Streamlit page
st.set_page_config(
    page_title="Strategy Fact Sheet",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour, then auto-refresh
def load_data():
    """Load all data from the database (PostgreSQL primary, SQLite local fallback)"""
    
    # Check if we have a database connection string (indicates cloud deployment)
    db_url = get_db_connection()
    
    # Add database type to cache key to prevent cross-contamination
    if db_url and POSTGRES_AVAILABLE:
        cache_key = f"postgres_{hash(db_url) % 10000}"
    else:
        cache_key = "sqlite_local"
    
    # Force cache invalidation if database type changes
    if 'last_db_type' not in st.session_state:
        st.session_state.last_db_type = cache_key
    elif st.session_state.last_db_type != cache_key:
        st.cache_data.clear()
        st.session_state.last_db_type = cache_key
    
    if db_url and POSTGRES_AVAILABLE:
        try:
            return load_data_postgres()
        except Exception as e:
            st.error(f"❌ PostgreSQL connection failed: {str(e)}")
            st.error("Please check your database connection in Streamlit secrets.")
            # Don't fallback in cloud - show the error instead
            raise e
    elif POSTGRES_AVAILABLE:
        # Try local PostgreSQL if secrets exist locally
        try:
            return load_data_postgres()
        except Exception as e:
            st.warning(f"PostgreSQL unavailable: {str(e)}. Using local SQLite.")
            return load_data_sqlite()
    else:
        # No PostgreSQL available, use SQLite
        return load_data_sqlite()

def get_db_connection():
    """Get database connection string from secrets"""
    try:
        # Streamlit Cloud secrets (primary for deployment)
        if hasattr(st, 'secrets') and 'db_url' in st.secrets:
            return st.secrets['db_url']
        
        # Local development with secrets.toml
        try:
            import toml
            if os.path.exists('secrets.toml'):
                secrets = toml.load('secrets.toml')
                db_url = secrets.get('db_url')
                if db_url:
                    return db_url
        except ImportError:
            pass
        
        return None
    except Exception as e:
        st.error(f"Error accessing secrets: {e}")
        return None

def load_data_postgres():
    """Load data from PostgreSQL database"""
    db_url = get_db_connection()
    if not db_url:
        raise Exception("No PostgreSQL connection string found in secrets")
    
    try:
        engine = create_engine(db_url, connect_args={"sslmode": "require"})
        
        # Test connection first
        with engine.connect() as test_conn:
            test_conn.execute(text("SELECT 1"))
        
        # Load all tables
        monthly_returns = pd.read_sql_query("SELECT * FROM monthly_returns ORDER BY date", engine)
        
        # Get most recent GA allocations (for current allocation display)
        current_gam_allocations = pd.read_sql_query("""
            SELECT * FROM gam_allocations 
            WHERE date = (SELECT MAX(date) FROM gam_allocations)
            ORDER BY asset_symbol
        """, engine)
        
        # Get historical allocations (for trailing 12M analysis)
        historical_allocations = pd.read_sql_query("SELECT * FROM gam_allocations ORDER BY date, asset_symbol", engine)
        
        # Get historical attribution (for attribution analysis)
        historical_attribution = pd.read_sql_query("SELECT * FROM gam_attribution ORDER BY date, asset_symbol", engine)
        
        benchmark_performance = pd.read_sql_query("SELECT * FROM benchmark_performance", engine)
        
        # Use all historical data for trailing 12 months view (will be filtered in chart functions)
        trailing_12m_allocations = historical_allocations.copy()
        trailing_12m_attribution = historical_attribution.copy()
        
        # Convert date columns
        for df in [monthly_returns, current_gam_allocations, historical_attribution, trailing_12m_allocations, trailing_12m_attribution]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        return {
            'monthly_returns': monthly_returns,
            'gam_allocations': current_gam_allocations,
            'gam_attribution': historical_attribution,
            'benchmark_performance': benchmark_performance,
            'trailing_12m_allocations': trailing_12m_allocations,
            'trailing_12m_attribution': trailing_12m_attribution
        }
        
    except Exception as e:
        st.error(f"Database query failed: {str(e)}")
        raise e
    finally:
        if 'engine' in locals():
            engine.dispose()

def load_data_sqlite():
    """Load data from SQLite database (fallback)"""
    if not os.path.exists('portfolio_data.db'):
        st.error("Database not found! Please run the migration script or ensure your database is available.")
        return create_empty_data_structure()
    
    conn = sqlite3.connect('portfolio_data.db')
    
    try:
        # Load all tables
        monthly_returns = pd.read_sql_query("SELECT * FROM monthly_returns ORDER BY date", conn)
        
        # Get most recent GA allocations (for current allocation display)
        current_gam_allocations = pd.read_sql_query("""
            SELECT * FROM gam_allocations 
            WHERE date = (SELECT MAX(date) FROM gam_allocations)
            ORDER BY asset_symbol
        """, conn)
        
        # Get historical allocations (for trailing 12M analysis)
        historical_allocations = pd.read_sql_query("SELECT * FROM gam_allocations ORDER BY date, asset_symbol", conn)
        
        # Get historical attribution (for attribution analysis)
        historical_attribution = pd.read_sql_query("SELECT * FROM gam_attribution ORDER BY date, asset_symbol", conn)
        
        benchmark_performance = pd.read_sql_query("SELECT * FROM benchmark_performance", conn)
        
        # Use historical data for trailing 12 months view (completed months only)
        trailing_12m_allocations = historical_allocations.copy()
        trailing_12m_attribution = historical_attribution.copy()
        
        # Convert date columns
        for df in [monthly_returns, current_gam_allocations, historical_attribution, trailing_12m_allocations, trailing_12m_attribution]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        return {
            'monthly_returns': monthly_returns,
            'gam_allocations': current_gam_allocations,
            'gam_attribution': historical_attribution,
            'benchmark_performance': benchmark_performance,
            'trailing_12m_allocations': trailing_12m_allocations,
            'trailing_12m_attribution': trailing_12m_attribution
        }
        
    finally:
        conn.close()

def create_empty_data_structure():
    """Create empty data structure for when database is not available"""
    empty_df = pd.DataFrame()
    return {
        'monthly_returns': empty_df,
        'gam_allocations': empty_df,
        'gam_attribution': empty_df,
        'benchmark_performance': empty_df,
        'trailing_12m_allocations': empty_df,
        'trailing_12m_attribution': empty_df
    }

def create_growth_chart(data):
    """Create Growth of $100,000 chart showing portfolio vs benchmarks"""
    df = data['monthly_returns'].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate cumulative growth starting from $100,000
    initial_investment = 100000
    
    # GA strategy growth
    df['ga_cumulative'] = initial_investment * (1 + df['gam_returns_net']).cumprod()
    
    # Use actual 60/40 portfolio returns from database instead of approximation
    df['benchmark_60_40'] = initial_investment * (1 + df['portfolio_60_40']).cumprod()
    
    fig = go.Figure()
    
    # Add GA strategy line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ga_cumulative'],
        mode='lines',
        name='RQA Global Adaptive',
        line=dict(color='#1e3a8a', width=3),  # Dark blue
        hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))

    # Add Global 60/40 benchmark
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['benchmark_60_40'],
        mode='lines',
        name='Global 60/40',
        line=dict(color='#6b7280', width=2),  # Grey
        hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))    # Create dynamic date range for title
    start_date = df['date'].min().strftime('%m/%d/%Y')
    end_date = df['date'].max().strftime('%m/%d/%Y')
    
    fig.update_layout(
        title={
            'text': f'GROWTH OF $100,000<br><sub>({start_date} - {end_date})</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#0f1419'}
        },
        xaxis_title='',
        yaxis_title='Portfolio Value ($)',
        height=450,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            tickformat='$,.0f',
            gridcolor='lightgray'
        ),
        xaxis=dict(
            gridcolor='lightgray',
            tickangle=0
        ),
        margin=dict(l=50, r=50, t=80, b=100)
    )
    
    return fig

def create_trailing_12m_allocations_chart(data):
    """Create Trailing 12 Month Allocations stacked bar chart"""
    # Use trailing_12m_allocations data which has the proper historical monthly data
    df = data['trailing_12m_allocations'].copy()
    
    if df.empty:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No allocation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title='Trailing 12 Month Allocations',
            xaxis_title='Date',
            yaxis_title='Allocation (%)',
            height=500
        )
        return fig
    
    # Filter for exactly trailing 12 months from the latest available month-end
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    
    # Handle case where latest_date is NaT (no valid dates)
    if pd.isna(latest_date):
        # If no valid dates, just use all available data
        pass
    else:
        # Get the latest month-year period
        latest_period = pd.Period(latest_date, freq='M')
        
        # Calculate exactly 12 months back from the latest period
        # For Sept 2025: Include Oct 2024, Nov 2024, ..., Aug 2025, Sep 2025 (12 months)
        periods_to_include = [latest_period - i for i in range(11, -1, -1)]  # Reverse order for chronological
        
        # Filter data to only include these specific month-year periods
        df['period'] = df['date'].dt.to_period('M')
        df = df[df['period'].isin(periods_to_include)]
    
    # Group by period and asset to get one entry per month-year combination
    monthly_df = df.groupby(['period', 'asset_name'])['allocation_percentage'].last().reset_index()
    
    # Convert period back to month_end for display and sort chronologically
    monthly_df['month_end'] = monthly_df['period'].dt.end_time
    monthly_df = monthly_df.sort_values('period')
    
    # Pivot data for stacked bar chart
    pivot_df = monthly_df.pivot(index='month_end', columns='asset_name', values='allocation_percentage')
    
    # Fill NaN values with 0
    pivot_df = pivot_df.fillna(0)
    
    # Convert to percentages (historical data is stored as decimals, so multiply by 100)
    pivot_df = pivot_df * 100
    
    # Create month labels for x-axis
    pivot_df.index = pivot_df.index.strftime('%b %Y')
    
    # Create stacked bar chart
    fig = go.Figure()
    
    colors = {
        'SP500': '#80c1ff',  
        'Gold': '#4da6ff',   
        'US_REITs': '#337fcc',  
        'US_LT_Treas': '#1a5fb4', 
        'Intl_Dev_Stocks': '#00509e', 
        'EM_Stocks': '#004080', 
        'Commodities': '#003366', 
        'Cash': '#001f3f' 
    }
    
    # Sort columns to match the stacking order from the screenshots
    column_order = ['Commodities', 'EM_Stocks', 'Gold', 'Intl_Dev_Stocks', 'SP500', 'US_LT_Treas', 'US_REITs', 'Cash']
    available_columns = [col for col in column_order if col in pivot_df.columns]
    other_columns = [col for col in pivot_df.columns if col not in column_order]
    final_columns = available_columns + other_columns
    
    # Create asset name mapping for consistent display names
    asset_name_map = {
        'SPDR Portfolio S&P 500 ETF': 'S&P 500',
        'iShares Gold Trust': 'Gold',
        'Vanguard FTSE Developed Markets ETF': 'Intl. Dev. Stocks',
        'Vanguard Real Estate Index Fund': 'U.S. REITs',
        'Schwab Emerging Markets Equity ETF': 'EM Stocks',
        'iShares GSCI Commodity Dynamic Roll Strategy ETF': 'Commodities',
        'iShares 20+ Year Treasury Bond ETF': 'U.S. LT Treas',
        'Cash': 'Cash',
        'SP500': 'S&P 500',
        'US_REITs': 'U.S. REITs', 
        'US_LT_Treas': 'U.S. LT Treas',
        'Intl_Dev_Stocks': 'Intl. Dev. Stocks',
        'EM_Stocks': 'EM Stocks',
        'Commodities': 'Commodities',
        'Gold': 'Gold'
    }

    for asset in final_columns:
        display_name = asset_name_map.get(asset, asset)
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=pivot_df[asset],
            name=display_name,
            marker_color=colors.get(asset, '#1f77b4'),
            hovertemplate=f'<b>{display_name}</b><br>Month: %{{x}}<br>Allocation: %{{y:.1f}}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Trailing 12 Month Allocations',
        xaxis_title='Month',
        yaxis_title='Allocation (%)',
        barmode='stack',  # Enable stacked bars
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(range=[0, 100]),  # Set y-axis to show 0-100%
        xaxis=dict(tickangle=45)  # Rotate month labels for better readability
    )
    
    return fig

def create_monthly_returns_chart(data):
    """Create Monthly Returns chart with rolling volatility"""
    df = data['monthly_returns'].copy()
    
    # Calculate rolling volatility (12-month)
    df['rolling_volatility'] = df['gam_returns_net'].rolling(window=12).std() * np.sqrt(12)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Returns (%)', 'Rolling 12-Month Volatility (%)'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Monthly returns bar chart
    colors = ['green' if x >= 0 else 'red' for x in df['gam_returns_net']]
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['gam_returns_net'] * 100,
            name='Monthly Returns',
            marker_color=colors,
            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Volatility line chart
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['rolling_volatility'] * 100,
            mode='lines',
            name='Volatility',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="GA Portfolio Performance Analysis"
    )
    
    return fig

def create_benchmark_performance_chart(data):
    """Create Benchmark Performance comparison with blue theme and professional styling"""
    df = data['benchmark_performance'].copy()
    
    # Portfolio-specific colors (consistent across all charts)
    portfolio_colors = {
        'GAM': "#1e3a8a",      # Dark blue for GAM
        '60/40': "#15803d",    # Dark green for 60/40
        '70/30': "#6b7280"     # Gray for 70/30
    }
    
    # Time period colors for the returns chart legend
    time_period_colors = ["#0B3C7D", "#1D5BBF", "#3C82F6", "#93B4FF"]  # YTD, 1Y, 5Y, SI
    
    # Create 2x2 subplots with proper spacing
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Returns Comparison (%)', 'Risk Metrics', 'Risk-Adjusted Returns', 'Beta to S&P 500'),
        horizontal_spacing=0.12,
        vertical_spacing=0.18
    )
    
    portfolios = df['portfolio'].tolist()
    
    # Map portfolio names for display (GAM -> GA)
    portfolio_display_names = ['GA' if p == 'GAM' else p for p in portfolios]
    
    # Returns comparison (grouped bars) - only these show in legend
    metrics_data = [
        ('ytd', 'YTD', time_period_colors[0]),
        ('one_year', '1Y', time_period_colors[1]),
        ('five_year', '5Y', time_period_colors[2]),
        ('since_inception', 'Since Inception', time_period_colors[3])
    ]
    
    for i, (metric, label, color) in enumerate(metrics_data):
        values = df[metric] * 100
        fig.add_trace(
            go.Bar(
                x=portfolio_display_names,
                y=values,
                name=label,
                marker_color=color,
                text=[f'{v:.1f}%' for v in values],
                textposition='outside',
                textfont=dict(color=color, size=11),
                showlegend=True,  # Only returns traces show in legend
                hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.2f}%<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Risk metrics (Standard Deviation) - portfolio-specific colors
    risk_values = df['standard_deviation'] * 100
    portfolio_risk_colors = [portfolio_colors[portfolio] for portfolio in portfolios]
    fig.add_trace(
        go.Bar(
            x=portfolio_display_names,
            y=risk_values,
            name='Volatility',
            marker_color=portfolio_risk_colors,
            text=[f'{v:.1f}%' for v in risk_values],
            textposition='outside',
            textfont=dict(color='#374151', size=11),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Volatility: %{y:.2f}%<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Sharpe Ratio - portfolio-specific colors
    sharpe_values = df['sharpe_ratio']
    portfolio_sharpe_colors = [portfolio_colors[portfolio] for portfolio in portfolios]
    fig.add_trace(
        go.Bar(
            x=portfolio_display_names,
            y=sharpe_values,
            name='Sharpe',
            marker_color=portfolio_sharpe_colors,
            text=[f'{v:.2f}' for v in sharpe_values],
            textposition='outside',
            textfont=dict(color='#374151', size=11),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Sharpe Ratio: %{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Beta to S&P 500 - portfolio-specific colors
    beta_values = df['beta_to_sp500']
    portfolio_beta_colors = [portfolio_colors[portfolio] for portfolio in portfolios]
    fig.add_trace(
        go.Bar(
            x=portfolio_display_names,
            y=beta_values,
            name='Beta',
            marker_color=portfolio_beta_colors,
            text=[f'{v:.2f}' for v in beta_values],
            textposition='outside',
            textfont=dict(color='#374151', size=11),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Beta: %{y:.2f}<extra></extra>"
        ),
        row=2, col=2
    )
    
    # Apply styling and layout
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        title_text="Portfolio Benchmark Performance Comparison",
        title_font=dict(size=18),
        
        # Global legend above the charts
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.05,
            yanchor="bottom",
            title=""
        ),
        
        # Layout margins
        margin=dict(l=60, r=30, t=60, b=60),
        
        # Grouped bar settings for returns chart
        barmode="group",
        bargap=0.18,
        bargroupgap=0.12,
        
        # Uniform text settings - allow text to show
        uniformtext_minsize=8,
        uniformtext_mode="show"
    )
    
    # Update y-axes with fixed ranges, titles, and styling
    fig.update_yaxes(
        range=[0, 18],  # Increased range to accommodate text labels above bars
        title_text="Returns (%)",
        tickformat=".1f",
        gridcolor="rgba(15,23,42,0.08)",
        showgrid=True,
        zeroline=False,
        row=1, col=1
    )
    
    fig.update_yaxes(
        range=[8, 14], 
        title_text="Volatility (%)",
        tickformat=".1f",
        gridcolor="rgba(15,23,42,0.08)",
        showgrid=True,
        zeroline=False,
        row=1, col=2
    )
    
    fig.update_yaxes(
        range=[0.5, 1.0], 
        title_text="Sharpe Ratio",
        tickformat=".2f",
        gridcolor="rgba(15,23,42,0.08)",
        showgrid=True,
        zeroline=False,
        row=2, col=1
    )
    
    fig.update_yaxes(
        range=[0.3, 0.9], 
        title_text="Beta",
        tickformat=".2f",
        gridcolor="rgba(15,23,42,0.08)",
        showgrid=True,
        zeroline=False,
        row=2, col=2
    )
    
    # Update x-axes to remove grid lines
    fig.update_xaxes(showgrid=False, zeroline=False)
    
    # Ensure labels don't clip and handle text positioning
    fig.update_traces(
        cliponaxis=False,
        selector=dict(type="bar")
    )
    
    # Style subplot titles
    fig.update_annotations(font=dict(size=14))
    
    return fig

def create_attribution_chart(data):
    """Create Portfolio Attribution stacked bar chart"""
    df = data['trailing_12m_attribution'].copy()
    
    if df.empty:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No attribution data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title='Trailing 12 Month Portfolio Attribution',
            xaxis_title='Date',
            yaxis_title='Attribution (%)',
            height=500
        )
        return fig
    
    # Filter for trailing 12 months using precise month-year periods
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    
    # Handle case where latest_date is NaT (no valid dates)
    if pd.isna(latest_date):
        # If no valid dates, just use all available data
        pass
    else:
        # Get the latest month-year period
        latest_period = pd.Period(latest_date, freq='M')
        
        # Calculate exactly 12 months back from the latest period (including current month)
        periods_to_include = [latest_period - i for i in range(12)]
        
        # Filter data to only include these specific month-year periods
        df['period'] = df['date'].dt.to_period('M')
        df = df[df['period'].isin(periods_to_include)]
    
    # Create month labels for x-axis using the period
    df['month_year'] = df['period'].dt.strftime('%b')
    
    # Group by period and asset to get one entry per month-year combination
    pivot_df = df.groupby(['month_year', 'asset_name'])['attribution_value'].sum().reset_index()
    pivot_df = pivot_df.pivot(index='month_year', columns='asset_name', values='attribution_value')
    pivot_df = pivot_df.fillna(0)
    
    # Convert to percentages 
    # Attribution data is stored as decimals (e.g., 0.00281 = 0.281%)
    # So multiply by 100 to get percentage values for display
    pivot_df = pivot_df * 100
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Color palette for assets (matching the allocation chart)
    colors = {
        'SP500': '#80c1ff',  
        'Gold': '#4da6ff',   
        'US_REITs': '#337fcc',  
        'US_LT_Treas': '#1a5fb4', 
        'Intl_Dev_Stocks': '#00509e', 
        'EM_Stocks': '#004080', 
        'Commodities': '#003366', 
        'Cash': '#001f3f' 
    }
    
    # Order columns same as allocation chart
    column_order = ['Commodities', 'EM_Stocks', 'Gold', 'Intl_Dev_Stocks', 'SP500', 'US_LT_Treas', 'US_REITs', 'Cash']
    available_columns = [col for col in column_order if col in pivot_df.columns]
    other_columns = [col for col in pivot_df.columns if col not in column_order]
    final_columns = available_columns + other_columns
    
    # Get the actual periods and sort them, then extract month abbreviations
    # For trailing 12 months, we want the most recent 12 months in chronological order
    if not df.empty:
        df_sorted = df.groupby('period')['month_year'].first().reset_index()
        df_sorted = df_sorted.sort_values('period')
        
        # Take only the last 12 months (or all available if less than 12)
        last_12_months = df_sorted.tail(12)
        ordered_months = last_12_months['month_year'].tolist()
        
        # Reindex pivot_df with the properly sorted months (last 12 months chronologically)
        pivot_df = pivot_df.reindex(ordered_months)
    else:
        ordered_months = list(pivot_df.index)
    
    for asset in final_columns:
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=pivot_df[asset],
            name=asset.replace('_', ' ').replace('SP500', 'S&P 500').replace('US LT Treas', 'U.S. LT Treas').replace('US REITs', 'U.S. REITs').replace('Intl Dev Stocks', 'Intl. Dev. Stocks').replace('EM Stocks', 'EM Stocks'),
            marker_color=colors.get(asset, '#1f77b4'),
            hovertemplate=f'<b>{asset}</b><br>Month: %{{x}}<br>Attribution: %{{y:.2f}}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Trailing 12 Month Portfolio Attribution',
        xaxis_title='Month',
        yaxis_title='Attribution (%)',
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        barmode='relative',  # This creates the stacked effect with positive and negative values
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Add horizontal line at zero
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    
    return fig

def calculate_ga_performance_metrics(data_dict):
    """Get GA performance metrics from database and calculate 3-year return"""
    benchmark_data = data_dict['benchmark_performance']
    monthly_returns_df = data_dict['monthly_returns']
    
    # Get GAM performance metrics from database
    gam_benchmark = benchmark_data[benchmark_data['portfolio'] == 'GAM']
    
    if not gam_benchmark.empty:
        gam_data = gam_benchmark.iloc[0]
        
        # Calculate 3-year return from monthly returns data
        df = monthly_returns_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        current_date = df['date'].max()
        three_years_ago = current_date - timedelta(days=3*365.25)
        three_year_data = df[df['date'] >= three_years_ago]
        
        if len(three_year_data) >= 24:  # Ensure we have at least 2 years of data
            three_year_total_return = (1 + three_year_data['gam_returns_net']).prod() - 1
            years = len(three_year_data) / 12
            three_year_annualized = (1 + three_year_total_return) ** (1/years) - 1 if years > 0 else 0
        else:
            three_year_annualized = 0.0
        
        return {
            'ytd': gam_data['ytd'],
            'one_year': gam_data['one_year'],
            'three_year': three_year_annualized,  # Calculated from monthly returns
            'five_year': gam_data['five_year'],
            'since_inception': gam_data['since_inception'],
            'standard_deviation': gam_data['standard_deviation'],
            'sharpe_ratio': gam_data['sharpe_ratio'],
            'beta_to_sp500': gam_data['beta_to_sp500'],
            'max_drawdown': 0.0,  # Not stored in database yet, could be added later
            'downside_deviation': 0.0  # Not stored in database yet, could be added later
        }
    else:
        # Fallback to default values if no data found
        return {
            'ytd': 0.0,
            'one_year': 0.0,
            'three_year': 0.0,
            'five_year': 0.0,
            'since_inception': 0.0,
            'standard_deviation': 0.0,
            'sharpe_ratio': 0.0,
            'beta_to_sp500': 0.0,
            'max_drawdown': 0.0,
            'downside_deviation': 0.0
        }

def create_allocation_pie_chart(data):
    """Create current allocation pie chart"""
    df = data['gam_allocations'].copy()  # Use current GAM allocations
    
    if df.empty:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No allocation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title='Current Asset Allocation',
            height=500
        )
        return fig
    
    # Get latest allocation
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    
    # Create asset name mapping for cleaner display
    asset_name_map = {
        'SPDR Portfolio S&P 500 ETF': 'U.S. Stocks (S&P 500)',
        'iShares Gold Trust': 'Gold',
        'Vanguard FTSE Developed Markets ETF': 'International Developed Stocks',
        'Vanguard Real Estate Index Fund': 'U.S. Real Estate (REITs)',
        'Schwab Emerging Markets Equity ETF': 'Emerging Market Stocks',
        'iShares GSCI Commodity Dynamic Roll Strategy ETF': 'Commodities',
        'iShares 20+ Year Treasury Bond ETF': 'U.S. Long Term Treasuries',
        'Cash': 'Cash',
        'SP500': 'U.S. Stocks (S&P 500)',
        'US_REITs': 'U.S. Real Estate (REITs)', 
        'US_LT_Treas': 'U.S. Long Term Treasuries',
        'Intl_Dev_Stocks': 'International Developed Stocks',
        'EM_Stocks': 'Emerging Market Stocks',
        'Commodities': 'Commodities',
        'Gold': 'Gold'
    }
    
    # Handle case where latest_date is NaT (no valid dates)
    if pd.isna(latest_date):
        date_str = "Latest Available"
        latest_allocation = df.iloc[-len(df.groupby('asset_name')):]  # Get last entry for each asset
    else:
        date_str = latest_date.strftime("%B %Y")
        latest_allocation = df[df['date'] == latest_date]
    
    # Apply name mapping
    latest_allocation['display_name'] = latest_allocation['asset_name'].map(asset_name_map).fillna(latest_allocation['asset_name'])
    
    # Custom blue color scheme for pie chart
    colors = {
        'SP500': '#80c1ff',  
        'Gold': '#4da6ff',   
        'US_REITs': '#337fcc',  
        'US_LT_Treas': '#1a5fb4', 
        'Intl_Dev_Stocks': '#00509e', 
        'EM_Stocks': '#004080', 
        'Commodities': '#003366', 
        'Cash': '#001f3f',
        # Map display names to colors as well
        'U.S. Stocks (S&P 500)': '#80c1ff',
        'Gold': '#4da6ff',
        'U.S. Real Estate (REITs)': '#337fcc',
        'U.S. Long Term Treasuries': '#1a5fb4',
        'International Developed Stocks': '#00509e',
        'Emerging Market Stocks': '#004080',
        'Commodities': '#003366',
        'Cash': '#001f3f'
    }
    
    # Create color list based on display names in the data
    pie_colors = [colors.get(name, '#1f77b4') for name in latest_allocation['display_name']]
    
    fig = px.pie(
        latest_allocation,
        values='allocation_percentage',
        names='display_name',
        title=f'Current Asset Allocation (as of {date_str})',
        color_discrete_sequence=pie_colors
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Allocation: %{percent}<extra></extra>'
    )
    
    fig.update_layout(height=500)
    
    return fig

# Main dashboard
def main():
    # Page config
    st.set_page_config(
        page_title="Strategy Fact Sheet",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for RQA blue theme and fact sheet styling
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    
    /* RQA Blue Color Scheme */
    :root {
        --rqa-blue: #0f1419;
        --rqa-light-blue: #1e3a8a;
        --rqa-dark-blue: #0f172a;
        --rqa-accent: #3b82f6;
        --rqa-text: #0f172a;
    }
    
    /* Header styling */
    .fact-sheet-header {
        background: linear-gradient(90deg, var(--rqa-blue) 0%, var(--rqa-light-blue) 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .strategy-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }
    
    .strategy-subtitle {
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Fact sheet sections */
    .fact-section {
        background: white;
        border: 2px solid var(--rqa-accent);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .fact-section h3 {
        color: var(--rqa-blue);
        border-bottom: 2px solid var(--rqa-accent);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .performance-metric {
        background: linear-gradient(135deg, var(--rqa-light-blue), var(--rqa-accent));
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
    }
    
    /* Custom tabs - Pill design with better styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #f8fafc;
        border-radius: 50px;
        padding: 8px 16px;
        border: 2px solid #e2e8f0;
        justify-content: center;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        color: #64748b;
        border-radius: 25px;
        font-weight: 500;
        padding: 0 24px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
        font-size: 0.95rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b5998;
        color: white;
        border: 1px solid #3b5998;
        box-shadow: 0 2px 8px rgba(59, 89, 152, 0.3);
    }
    
    /* Key facts styling */
    .key-facts {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        border-left: 4px solid var(--rqa-blue);
        padding: 1rem;
        border-radius: 4px;
    }
    
    .section-header {
        color: var(--rqa-blue);
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--rqa-accent);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    # Calculate key metrics for display
    ga_metrics = calculate_ga_performance_metrics(data)
    
    # Add logo to sidebar above navigation
    st.sidebar.image("logo.png", width=200)
    st.sidebar.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # Main navigation choice
    view_option = st.sidebar.selectbox(
        "Navigation",
        ["Strategy Fact Sheet", "Interactive Analytics"],
        index=0
    )
    
    if view_option == "Strategy Fact Sheet":
        create_fact_sheet_landing(data, ga_metrics)
    else:
        create_analytics_dashboard(data, ga_metrics)

def generate_pdf_report(data, ga_metrics):
    """Generate PDF report of the strategy fact sheet"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.linecharts import HorizontalLineChart
        import tempfile
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.HexColor('#1e3a8a')
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#3b5998'),
            backColor=colors.HexColor('#f0f4f8'),
            borderPadding=8
        )
        
        story = []
        
        # Title
        story.append(Paragraph("RQA Global Adaptive Strategy", title_style))
        story.append(Paragraph(f"Fact Sheet - {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Strategy Overview
        story.append(Paragraph("STRATEGY OVERVIEW", header_style))
        overview_text = """
        Global Adaptive (GA) is designed to meet the challenges of today's ever-changing markets by allocating 
        capital across major global asset classes through a disciplined, data-driven process. Traditional approaches, 
        such as the 60/40 stock-bond portfolio, rely on static assumptions that may leave investors overexposed when 
        conditions shift. Instead, GA takes a dynamic approach. By continuously evaluating global opportunities, the 
        strategy seeks to increase exposure to asset classes showing strength while reducing or avoiding those losing 
        momentum. This adaptability allows the portfolio to respond proactively rather than reactively, providing the 
        potential for stronger risk-adjusted returns over time.
        """
        story.append(Paragraph(overview_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Performance Metrics
        story.append(Paragraph("STRATEGY RETURNS (Net of Fees)", header_style))
        
        # Get benchmark data for 60/40 portfolio
        benchmark_data = data.get('benchmark_performance', pd.DataFrame())
        if not benchmark_data.empty:
            # Find 60/40 portfolio data
            sixtyforty_data = benchmark_data[benchmark_data['portfolio'] == '60/40']
            if not sixtyforty_data.empty:
                sixtyforty_metrics = sixtyforty_data.iloc[0]
                sixtyforty_ytd = f"{sixtyforty_metrics.get('ytd', 0)*100:.2f}%"
                sixtyforty_1y = f"{sixtyforty_metrics.get('one_year', 0)*100:.2f}%"
                sixtyforty_5y = f"{sixtyforty_metrics.get('five_year', 0)*100:.2f}%"
                sixtyforty_si = f"{sixtyforty_metrics.get('since_inception', 0)*100:.2f}%"
            else:
                sixtyforty_ytd = sixtyforty_1y = sixtyforty_5y = sixtyforty_si = "N/A"
        else:
            sixtyforty_ytd = sixtyforty_1y = sixtyforty_5y = sixtyforty_si = "N/A"
        
        # Create performance table
        perf_data = [
            ['Period', 'RQA Global Adaptive', 'Global 60/40'],
            ['YTD', f"{ga_metrics['ytd']*100:.2f}%", sixtyforty_ytd],
            ['1 Year', f"{ga_metrics['one_year']*100:.2f}%", sixtyforty_1y],
            ['5 Year', f"{ga_metrics['five_year']*100:.2f}%", sixtyforty_5y],
            ['Since Inception', f"{ga_metrics['since_inception']*100:.2f}%", sixtyforty_si]
        ]
        
        perf_table = Table(perf_data)
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b5998')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(perf_table)
        story.append(Spacer(1, 20))
        
        # Key Metrics
        story.append(Paragraph("KEY METRICS", header_style))
        metrics_data = [
            ['Metric', 'Value'],
            ['Sharpe Ratio', f"{ga_metrics.get('sharpe_ratio', 0):.2f}"],
            ['Standard Deviation', f"{ga_metrics.get('standard_deviation', 0)*100:.1f}%"],
            ['Beta to S&P 500', f"{ga_metrics.get('beta_to_sp500', 0):.2f}"]
        ]
        
        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b5998')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Key Strategy Features
        story.append(Paragraph("KEY STRATEGY FEATURES", header_style))
        features_text = """
        • Dynamic Asset Allocation across global markets<br/>
        • Quantitative risk management framework<br/>
        • Tactical rebalancing based on market conditions<br/>
        • Diversified exposure to multiple asset classes<br/>
        • Active downside protection during market stress<br/>
        • Evidence-based investment process
        """
        story.append(Paragraph(features_text, styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Richmond Quantitative Advisors"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            alignment=1,  # Center alignment
            textColor=colors.grey
        )
        story.append(Paragraph(footer_text, footer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        raise Exception("ReportLab library not installed. Please run: pip install reportlab")
    except Exception as e:
        raise Exception(f"Error generating PDF: {str(e)}")

def create_fact_sheet_landing(data, ga_metrics):
    # Clean Professional Header with Logo and Strategy Name
    import base64
    import os
    
    # Read and encode logo
    logo_html = ""
    if os.path.exists("logo.png"):
        with open("logo.png", "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{logo_data}" style="width: auto; height: 100px; border-radius: 12px; background: white; padding: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);"/>'
    
    # Create the professional header with logo on left and strategy name on right
    
    # Read and encode globe image for background
    globe_bg = ""
    if os.path.exists("globe.png"):
        with open("globe.png", "rb") as f:
            globe_data = base64.b64encode(f.read()).decode()
            globe_bg = f"url('data:image/png;base64,{globe_data}')"
    
    st.markdown(f"""
    <div style="
        background: 
            linear-gradient(135deg, rgba(37, 99, 235, 0.75) 0%, rgba(59, 130, 246, 0.75) 50%, rgba(96, 165, 250, 0.75) 100%),
            {globe_bg};
        background-size: cover, contain;
        background-position: center, center;
        background-repeat: no-repeat;
        color: white;
        padding: 3rem 4rem;
        margin: -1rem -1rem 3rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        min-height: 140px;
        position: relative;
    ">
        <div style="flex-shrink: 0;">
            {logo_html}
        </div>
        <div style="text-align: center; flex: 1; margin-left: 2rem;">
            <h1 style="
                margin: 0;
                font-size: 4rem;
                font-weight: 300;
                letter-spacing: 0.2rem;
                color: white;
                line-height: 1.1;
                text-shadow: 0 3px 6px rgba(0,0,0,0.4);
            ">GLOBAL</h1>
            <h1 style="
                margin: -0.2rem 0 0.3rem 0;
                font-size: 4rem;
                font-weight: 300;
                letter-spacing: 0.2rem;
                color: white;
                line-height: 1.1;
                text-shadow: 0 3px 6px rgba(0,0,0,0.4);
            ">ADAPTIVE</h1>
            <div style="width: 80%; height: 2px; background: white; margin: 1rem auto;"></div>
            <h2 style="
                margin: 0;
                font-size: 2.5rem;
                font-weight: 300;
                letter-spacing: 0.3rem;
                color: white;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            ">STRATEGY</h2>
        </div>
        <div style="flex-shrink: 0; width: 120px;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Professional PDF Export Icon - positioned above everything in top right
    try:
        pdf_buffer = generate_pdf_report(data, ga_metrics)
        
        # Create download link with professional icon
        b64_pdf = base64.b64encode(pdf_buffer).decode()
        filename = f"RQA_Global_Adaptive_Strategy_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        st.markdown(f"""
        <div style="
            position: fixed; 
            top: 15px; 
            right: 25px; 
            z-index: 9999;
            background: transparent;
        ">
            <a href="data:application/pdf;base64,{b64_pdf}" 
               download="{filename}"
               style="text-decoration: none;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" 
                     style="cursor: pointer; opacity: 0.7; transition: opacity 0.2s;"
                     onmouseover="this.style.opacity='1'"
                     onmouseout="this.style.opacity='0.7'">
                    <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V8L14 2Z" 
                          fill="#1e3a8a"/>
                    <path d="M14 2V8H20" fill="#60a5fa"/>
                    <path d="M12 18L8 14H10.5V10H13.5V14H16L12 18Z" fill="white"/>
                </svg>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        # Fallback if PDF generation fails
        st.markdown(f"""
        <div style="
            position: fixed; 
            top: 15px; 
            right: 25px; 
            z-index: 9999;
        ">
            <div title="PDF Export Error: {str(e)}" style="opacity: 0.3;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="#cccccc">
                    <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V8L14 2Z"/>
                    <path d="M14 2V8H20" fill="#e0e0e0"/>
                    <path d="M12 18L8 14H10.5V10H13.5V14H16L12 18Z" fill="white"/>
                </svg>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Strategy Overview with consistent header format
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">STRATEGY OVERVIEW</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        Global Adaptive (GA) is designed to meet the challenges of today's ever-changing markets by allocating capital across major global asset classes through a disciplined, data-driven process. Traditional approaches, such as the 60/40 stock-bond portfolio, rely on static assumptions that may leave investors overexposed when conditions shift.
        
        Instead, GA takes a dynamic approach. By continuously evaluating global opportunities, the strategy seeks to increase exposure to asset classes showing strength while reducing or avoiding those losing momentum. This adaptability allows the portfolio to respond proactively rather than reactively, providing the potential for stronger risk-adjusted returns over time.
        
        Through this evidence-based framework, GA strives to deliver three key benefits: more consistent growth, enhanced stability, and improved capital preservation. The result is a portfolio designed not only to participate in market gains, but also to better withstand periods of stress — an advantage that can compound meaningfully for investors over the long run.
        """)
        
        # Growth Chart
        st.plotly_chart(create_growth_chart(data), use_container_width=True)
        
        # Strategy Returns with consistent header format
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">STRATEGY RETURNS (Net of Fees)</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create the performance table using database data
        benchmark_data = data['benchmark_performance'].copy()
        
        # Get Global 60/40 metrics from database
        global_6040 = benchmark_data[benchmark_data['portfolio'] == '60/40'].iloc[0] if not benchmark_data.empty else None
        
        # Calculate 3-year return for 60/40 from monthly returns data
        monthly_returns_df = data['monthly_returns']
        df = monthly_returns_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        current_date = df['date'].max()
        three_years_ago = current_date - timedelta(days=3*365.25)
        three_year_data = df[df['date'] >= three_years_ago]
        
        if len(three_year_data) >= 24:  # Ensure we have at least 2 years of data
            three_year_total_return_6040 = (1 + three_year_data['portfolio_60_40']).prod() - 1
            years = len(three_year_data) / 12
            three_year_annualized_6040 = (1 + three_year_total_return_6040) ** (1/years) - 1 if years > 0 else 0
        else:
            three_year_annualized_6040 = 0.0
        
        performance_data = {
            '': ['RQA Global Adaptive', 'Global 60/40'],
            'YTD': [f"{ga_metrics['ytd']*100:.1f}%", 
                   f"{global_6040['ytd']*100:.1f}%" if global_6040 is not None else "N/A"],
            '1 Year': [f"{ga_metrics['one_year']*100:.1f}%", 
                      f"{global_6040['one_year']*100:.1f}%" if global_6040 is not None else "N/A"],
            '3 Year': [f"{ga_metrics['three_year']*100:.1f}%", 
                      f"{three_year_annualized_6040*100:.1f}%"],  # Now using calculated 3-year return
            '5 Year': [f"{ga_metrics['five_year']*100:.1f}%", 
                      f"{global_6040['five_year']*100:.1f}%" if global_6040 is not None else "N/A"],
            'Since Inception': [f"{ga_metrics['since_inception']*100:.1f}%", 
                              f"{global_6040['since_inception']*100:.1f}%" if global_6040 is not None else "N/A"]
        }
        
        # Display as a clean table
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, hide_index=True, use_container_width=True)
        
        # Risk Management with consistent header format and table
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">RISK MANAGEMENT</h2>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1rem; font-style: italic; font-weight: 300;">"Risk Forward Framework"</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk metrics table using database data
        risk_data = {
            'Metric': ['Annualized Volatility', 'Sharpe Ratio (RF=0)', 'Beta (vs. S&P)'],
            'GA': [f"{ga_metrics['standard_deviation']*100:.1f}%", 
                   f"{ga_metrics['sharpe_ratio']:.2f}", 
                   f"{ga_metrics['beta_to_sp500']:.2f}"],
            'Global 60/40': [f"{global_6040['standard_deviation']*100:.1f}%" if global_6040 is not None else "N/A",
                            f"{global_6040['sharpe_ratio']:.2f}" if global_6040 is not None else "N/A",
                            f"{global_6040['beta_to_sp500']:.2f}" if global_6040 is not None else "N/A"]
        }
        
        df_risk = pd.DataFrame(risk_data)
        st.dataframe(df_risk, hide_index=True, use_container_width=True)
        
        # Portfolio Management with consistent header format
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">PORTFOLIO MANAGEMENT</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Format portfolio management content in a structured way
        mgmt_col1, mgmt_col2 = st.columns(2)
        
        with mgmt_col1:
            st.markdown("""
            **Richmond Quantitative Advisors**  
            [www.RichmondQuant.com](https://www.RichmondQuant.com)
            
            **Andrew S. Holpe**  
            [AHolpe@RichmondQuant.com](mailto:AHolpe@RichmondQuant.com)  
            *Portfolio Manager*
            """)
        
        with mgmt_col2:
            st.markdown("""
            
            
            **John D. Ellison, CFA**  
            [Jellison@RichmondQuant.com](mailto:Jellison@RichmondQuant.com)  
            *Portfolio Manager*
            """)
    
    with col2:
        # Key Facts with enhanced styling
        # Key Facts with consistent header format
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">KEY FACTS</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **INCEPTION DATE:**  
        July 2019
        
        **STYLE:**  
        Multi-Asset & Dynamic Global Allocation
        
        **BENCHMARK:**  
        Global 60/40 "60% ACWI / 40% AGG"
        
        **ASSET UNIVERSE:**  
        U.S. Equities, Developed & Emerging Market Equities, Global Real Estate Investment Trusts (REIT’s), Global Bonds, U.S. Treasuries, Precious Metals, and Global Commodity Composites
        
        **OFFERING VEHICLE:**  
        Separately Managed Accounts & Custom Model Delivery
        
        **INVESTOR AVAILABILITY:**  
        Unaccredited & Qualified
        
        **LIQUIDITY:**  
        Daily via ETF holdings
        """)
        
        # Current allocation section with consistent header format
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">CURRENT ALLOCATION</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Get current allocation data
        df = data['gam_allocations'].copy()
        
        if df.empty:
            st.markdown("*No allocation data available*")
        else:
            df['date'] = pd.to_datetime(df['date'])
            latest_date = df['date'].max()
            
            # Handle case where latest_date is NaT (no valid dates)
            if pd.isna(latest_date):
                latest_allocation = df.iloc[-len(df.groupby('asset_name')):][['asset_name', 'allocation_percentage']].copy()
            else:
                latest_allocation = df[df['date'] == latest_date][['asset_name', 'allocation_percentage']].copy()
                
            latest_allocation = latest_allocation.sort_values('allocation_percentage', ascending=False)
            
            # Create better allocation display with proper formatting and asset renaming
            asset_name_map = {
                'SP500': 'U.S. Stocks (S&P 500)',
                'US_REITs': 'U.S. Real Estate (REITs)', 
                'US_LT_Treas': 'U.S. Long Term Treasuries',
                'Intl_Dev_Stocks': 'International Developed Stocks',
                'EM_Stocks': 'Emerging Market Stocks',
                'Commodities': 'Commodities',
                'Gold': 'Gold',
                'Cash': 'Cash',
                # Full ETF name mappings
                'SPDR Portfolio S&P 500 ETF': 'U.S. Stocks (S&P 500)',
                'iShares Gold Trust': 'Gold',
                'Vanguard FTSE Developed Markets ETF': 'International Developed Stocks',
                'Vanguard Real Estate Index Fund': 'U.S. Real Estate (REITs)',
                'Schwab Emerging Markets Equity ETF': 'Emerging Market Stocks',
                'iShares GSCI Commodity Dynamic Roll Strategy ETF': 'Commodities',
                'iShares 20+ Year Treasury Bond ETF': 'U.S. Long Term Treasuries',
                # Additional possible variants
                'iShares Core MSCI Total International Stock ETF': 'International Developed Stocks',
                'Vanguard Real Estate Index Fund ETF Shares': 'U.S. Real Estate (REITs)',
                'Schwab Fundamental Emerging Markets Large Company Index Fund': 'Emerging Market Stocks',
                'Invesco DB Commodity Index Tracking Fund': 'Commodities'
            }
            
            for _, row in latest_allocation.iterrows():
                asset_name = row['asset_name']
                # Map to display name - check for exact match first, then partial matches
                display_name = asset_name_map.get(asset_name, asset_name)
                if display_name == asset_name:  # No exact match found, try partial matching
                    for key, value in asset_name_map.items():
                        if key.lower() in asset_name.lower() or asset_name.lower() in key.lower():
                            display_name = value
                            break
                
                allocation = row['allocation_percentage'] * 100  # Convert to percentage
                
                # Show more decimal places for small allocations (like cash)
                if allocation < 1.0:
                    allocation_str = f"{allocation:.2f}%"
                else:
                    allocation_str = f"{allocation:.1f}%"
                
                # Enhanced allocation display with progress bars
                bar_width = int(allocation * 2)  # Scale for visual effect
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                        <span style="font-weight: bold; color: #1e3a8a; font-size: 1.1rem;">{display_name}</span>
                        <span style="font-weight: bold; color: #3b5998; font-size: 1.1rem;">{allocation_str}</span>
                    </div>
                    <div style="width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden;">
                        <div style="width: {bar_width}%; height: 100%; background: linear-gradient(90deg, #3b5998, #60a5fa); transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
        # Key Strategy Features (standard header, icon bullets, no box outline)
        st.markdown("""
        <div style="background: #3b5998; color: white; padding: 0.9rem 1.4rem; border-radius: 8px; margin: 0.5rem 0 0.4rem 0;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">KEY STRATEGY FEATURES</h2>
        </div>
        <ul style="list-style:none; padding:0; margin:0 0 0.5rem 0;">
            <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem; color:#1f2937;">
                <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                <span style="line-height:1.15;">Tactical Asset Allocation</span>
            </li>
            <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem; color:#1f2937;">
                <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                <span style="line-height:1.15;">Global Diversification</span>
            </li>
            <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem; color:#1f2937;">
                <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                <span style="line-height:1.15;">Systematic Risk Management</span>
            </li>
            <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem; color:#1f2937;">
                <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                <span style="line-height:1.15;">Daily Liquidity</span>
            </li>
            <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0; font-size:0.95rem; color:#1f2937;">
                <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                <span style="line-height:1.15;">Monthly Rebalancing</span>
            </li>
        </ul>
        """, unsafe_allow_html=True)

        # Download factsheet button only
        pdf_buffer = generate_pdf_report(data, ga_metrics)
        
        st.download_button(
            label="📄 Download Factsheet",
            data=pdf_buffer,
            file_name="RQA_Strategy_Factsheet.pdf",
            mime="application/pdf",
            key="factsheet_download",
            help="Click to download the strategy factsheet PDF",
            use_container_width=True
        )

    # Add disclaimer with enhanced styling
    st.markdown("""
    <div style="margin-top: 2rem; 
                padding: 1rem; 
                background: linear-gradient(135deg, #f9fafb, #e5e7eb); 
                border-radius: 8px; 
                font-size: 0.85rem; 
                color: #6b7280; 
                border-left: 4px solid #9ca3af;">
        <strong>Important Disclosures:</strong> This presentation is for informational purposes only. Past performance does not guarantee future results. 
        All investments involve risk of loss. The strategy may not be suitable for all investors. Please consult with a financial advisor before investing.
        Performance figures are presented net of estimated management fees and related ETF fees and expenses.
    </div>
    """, unsafe_allow_html=True)

def create_analytics_dashboard(data, ga_metrics):
    # Clean Professional Header with Logo and Analytics Name
    import base64
    import os
    
    # Read and encode logo
    logo_html = ""
    if os.path.exists("logo.png"):
        with open("logo.png", "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{logo_data}" style="width: auto; height: 100px; border-radius: 12px; background: white; padding: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);"/>'
    
    # Create the professional header with logo on left and analytics name on right
    
    # Read and encode globe image for background
    globe_bg = ""
    if os.path.exists("globe.png"):
        with open("globe.png", "rb") as f:
            globe_data = base64.b64encode(f.read()).decode()
            globe_bg = f"url('data:image/png;base64,{globe_data}')"
    
    st.markdown(f"""
    <div style="
        background: 
            linear-gradient(135deg, rgba(37, 99, 235, 0.75) 0%, rgba(59, 130, 246, 0.75) 50%, rgba(96, 165, 250, 0.75) 100%),
            {globe_bg};
        background-size: cover, contain;
        background-position: center, center;
        background-repeat: no-repeat;
        color: white;
        padding: 3rem 4rem;
        margin: -1rem -1rem 3rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        min-height: 140px;
        position: relative;
    ">
        <div style="flex-shrink: 0;">
            {logo_html}
        </div>
        <div style="text-align: center; flex: 1; margin-left: 2rem;">
            <h1 style="
                margin: 0;
                font-size: 4rem;
                font-weight: 300;
                letter-spacing: 0.2rem;
                color: white;
                line-height: 1.1;
                text-shadow: 0 3px 6px rgba(0,0,0,0.4);
            "            ">PORTFOLIO</h1>
            <div style="width: 80%; height: 2px; background: white; margin: 1rem auto;"></div>
            <h2 style="
                    margin: 0;
                    font-size: 4rem;
                    font-weight: 300;
                    letter-spacing: 0.2rem;
                    color: white;
                    text-shadow: 0 3px 6px rgba(0,0,0,0.4);
                ">ANALYTICS</h2>
        </div>
        <div style="flex-shrink: 0; width: 120px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar metrics
    st.sidebar.header("Performance Metrics")
    st.sidebar.metric("YTD Return", f"{ga_metrics['ytd']*100:.2f}%")
    st.sidebar.metric("1-Year Return", f"{ga_metrics['one_year']*100:.2f}%")
    st.sidebar.metric("5-Year Return", f"{ga_metrics['five_year']*100:.2f}%")
    st.sidebar.metric("Since Inception", f"{ga_metrics['since_inception']*100:.2f}%")
    st.sidebar.metric("Sharpe Ratio", f"{ga_metrics['sharpe_ratio']:.3f}")
    st.sidebar.metric("Volatility", f"{ga_metrics['standard_deviation']*100:.2f}%")
    st.sidebar.metric("Beta to S&P 500", f"{ga_metrics['beta_to_sp500']:.3f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Current Allocation", 
        "Trailing 12M Allocations", 
        "Monthly Returns", 
        "Benchmark Performance", 
        "Attribution Analysis"
    ])
    
    with tab1:
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">CURRENT PORTFOLIO ALLOCATION</h2>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(create_allocation_pie_chart(data), use_container_width=True)
        
        with col2:
            # Show allocation table using current GAM data with abbreviated names
            df = data['gam_allocations'].copy()
            
            if df.empty:
                st.markdown("### Current Allocation Breakdown")
                st.markdown("*No allocation data available*")
            else:
                df['date'] = pd.to_datetime(df['date'])
                latest_date = df['date'].max()
                
                # Handle case where latest_date is NaT (no valid dates)
                if pd.isna(latest_date):
                    latest_allocation = df.iloc[-len(df.groupby('asset_name')):][['asset_name', 'allocation_percentage']].copy()
                else:
                    latest_allocation = df[df['date'] == latest_date][['asset_name', 'allocation_percentage']].copy()
                
                # Create asset name mapping for cleaner display
                asset_name_map = {
                    'SPDR Portfolio S&P 500 ETF': 'U.S. Stocks (S&P 500)',
                    'iShares Gold Trust': 'Gold',
                    'Vanguard FTSE Developed Markets ETF': 'International Developed Stocks',
                    'Vanguard Real Estate Index Fund': 'U.S. Real Estate (REITs)',
                    'Schwab Emerging Markets Equity ETF': 'Emerging Market Stocks',
                    'iShares GSCI Commodity Dynamic Roll Strategy ETF': 'Commodities',
                    'iShares 20+ Year Treasury Bond ETF': 'U.S. Long Term Treasuries',
                    'Cash': 'Cash',
                    'SP500': 'U.S. Stocks (S&P 500)',
                    'US_REITs': 'U.S. Real Estate (REITs)', 
                    'US_LT_Treas': 'U.S. Long Term Treasuries',
                    'Intl_Dev_Stocks': 'International Developed Stocks',
                    'EM_Stocks': 'Emerging Market Stocks',
                    'Commodities': 'Commodities',
                    'Gold': 'Gold'
                }
                
                # Apply name mapping
                latest_allocation['display_name'] = latest_allocation['asset_name'].map(asset_name_map).fillna(latest_allocation['asset_name'])
                
                # Create final dataframe with abbreviated names
                display_allocation = latest_allocation[['display_name', 'allocation_percentage']].copy()
                display_allocation = display_allocation.sort_values('allocation_percentage', ascending=False)
                display_allocation.columns = ['Asset Class', 'Allocation (%)']
                display_allocation['Allocation (%)'] = (display_allocation['Allocation (%)'] * 100).round(2)  # Convert to percentage
                
                st.markdown("### Current Allocation Breakdown")
                st.dataframe(display_allocation, width='stretch', hide_index=True)
    
    with tab2:
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">TRAILING 12 MONTH ALLOCATIONS</h2>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_trailing_12m_allocations_chart(data), use_container_width=True)
        
        st.markdown("**Analysis:** This stacked bar chart shows how the portfolio allocation has evolved over the trailing 12 months. You can see the dynamic rebalancing across different asset classes.")
    
    with tab3:
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">MONTHLY RETURNS & VOLATILITY</h2>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_monthly_returns_chart(data), use_container_width=True)
        
        # Summary statistics
        returns_data = data['monthly_returns']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Monthly Return", f"{returns_data['gam_returns_net'].mean()*100:.2f}%")
        with col2:
            st.metric("Best Month", f"{returns_data['gam_returns_net'].max()*100:.2f}%")
        with col3:
            st.metric("Worst Month", f"{returns_data['gam_returns_net'].min()*100:.2f}%")
        with col4:
            st.metric("Win Rate", f"{(returns_data['gam_returns_net'] > 0).mean()*100:.1f}%")
    
    with tab4:
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">BENCHMARK PERFORMANCE COMPARISON</h2>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_benchmark_performance_chart(data), use_container_width=True)
        
        
    
    with tab5:
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">PORTFOLIO ATTRIBUTION ANALYSIS</h2>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_attribution_chart(data), use_container_width=True)
        
        # Attribution summary
        attribution_data = data['trailing_12m_attribution']
        attribution_data['date'] = pd.to_datetime(attribution_data['date'])
        latest_date = attribution_data['date'].max()
        twelve_months_ago = latest_date - timedelta(days=365)
        attribution_data = attribution_data[attribution_data['date'] >= twelve_months_ago]
        attribution_summary = attribution_data.groupby('asset_name')['attribution_value'].sum() * 100
        attribution_summary = attribution_summary.sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top Contributors")
            top_contributors = attribution_summary.head(4)
            for asset, contrib in top_contributors.items():
                st.metric(asset, f"{contrib:.2f}%")
        
        with col2:
            st.markdown("### Detractors")
            detractors = attribution_summary.tail(4)
            for asset, contrib in detractors.items():
                st.metric(asset, f"{contrib:.2f}%")
    
if __name__ == "__main__":
    main()
