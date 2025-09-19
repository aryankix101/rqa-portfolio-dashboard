import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import os

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

@st.cache_data
def load_data():
    """Load all data from the database (PostgreSQL or SQLite fallback)"""
    
    # Try PostgreSQL first if available
    if POSTGRES_AVAILABLE:
        try:
            return load_data_postgres()
        except Exception as e:
            st.warning(f"PostgreSQL connection failed: {str(e)}. Using SQLite fallback.")
    
    # Fallback to SQLite
    return load_data_sqlite()

def get_db_connection():
    """Get database connection string from secrets"""
    try:
        # Streamlit Cloud secrets
        if hasattr(st, 'secrets') and 'db_url' in st.secrets:
            return st.secrets['db_url']
        
        # Local development with secrets.toml
        try:
            import toml
            if os.path.exists('secrets.toml'):
                secrets = toml.load('secrets.toml')
                return secrets.get('db_url')
        except ImportError:
            pass
        
        return None
    except Exception:
        return None

def load_data_postgres():
    """Load data from PostgreSQL database"""
    db_url = get_db_connection()
    if not db_url:
        raise Exception("No PostgreSQL connection string found")
    
    engine = create_engine(db_url, connect_args={"sslmode": "require"})
    
    try:
        # Load all tables
        monthly_returns = pd.read_sql_query("SELECT * FROM monthly_returns ORDER BY date", engine)
        
        # Get current GAM allocations
        current_gam_allocations = pd.read_sql_query("SELECT * FROM gam_allocations WHERE date >= '2025-09-01' ORDER BY date, asset_symbol", engine)
        
        # Get historical allocations
        historical_allocations = pd.read_sql_query("SELECT * FROM gam_allocations WHERE date <= '2025-07-31' ORDER BY date, asset_symbol", engine)
        
        # Get historical attribution
        historical_attribution = pd.read_sql_query("SELECT * FROM gam_attribution WHERE date <= '2025-07-31' ORDER BY date, asset_symbol", engine)
        
        benchmark_performance = pd.read_sql_query("SELECT * FROM benchmark_performance", engine)
        
        # Use historical data for trailing 12 months view
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
        
        # Get current GAM allocations (most recent daily data for current allocation display)
        current_gam_allocations = pd.read_sql_query("SELECT * FROM gam_allocations WHERE date >= '2025-09-01' ORDER BY date, asset_symbol", conn)
        
        # Get historical allocations (only completed months for trailing 12M analysis)
        historical_allocations = pd.read_sql_query("SELECT * FROM gam_allocations WHERE date <= '2025-07-31' ORDER BY date, asset_symbol", conn)
        
        # Get historical attribution (only completed months)
        historical_attribution = pd.read_sql_query("SELECT * FROM gam_attribution WHERE date <= '2025-07-31' ORDER BY date, asset_symbol", conn)
        
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
    
    # Use historical data for trailing 12 months view (completed months only)
    trailing_12m_allocations = historical_allocations.copy()
    trailing_12m_attribution = historical_attribution.copy()
    
    conn.close()
    
    # Convert date columns
    for df in [monthly_returns, current_gam_allocations, historical_attribution, trailing_12m_allocations, trailing_12m_attribution]:
        df['date'] = pd.to_datetime(df['date'])
    
    return {
        'monthly_returns': monthly_returns,
        'gam_allocations': current_gam_allocations,
        'gam_attribution': historical_attribution,
        'benchmark_performance': benchmark_performance,
        'trailing_12m_allocations': trailing_12m_allocations,
        'trailing_12m_attribution': trailing_12m_attribution
    }

def create_growth_chart(data):
    """Create Growth of $100,000 chart showing portfolio vs benchmarks"""
    df = data['monthly_returns'].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate cumulative growth starting from $100,000
    initial_investment = 100000
    
    # GA strategy growth
    df['ga_cumulative'] = initial_investment * (1 + df['gam_returns_net']).cumprod()
    
    # Create benchmark data (simplified for demo - in real implementation you'd have actual benchmark data)
    # For now, using estimated benchmark performance
    df['benchmark_60_40'] = initial_investment * (1 + df['gam_returns_net'] * 0.85).cumprod()  # Slightly lower returns
    
    fig = go.Figure()
    
    # Add GA strategy line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ga_cumulative'],
        mode='lines',
        name='RQA Global Adaptive',
        line=dict(color='#0f1419', width=3),
        hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add Global 60/40 benchmark
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['benchmark_60_40'],
        mode='lines',
        name='Global 60/40',
        line=dict(color='#16a085', width=2),
        hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'GROWTH OF $100,000<br><sub>(7/1/2019 - 9/18/2025)</sub>',
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
    
    # Filter for trailing 12 months from the last available month-end
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    twelve_months_ago = latest_date - timedelta(days=365)
    df = df[df['date'] >= twelve_months_ago]
    
    # Group by month-end dates to get one entry per month
    df['month_end'] = df['date'].dt.to_period('M').dt.end_time
    monthly_df = df.groupby(['month_end', 'asset_name'])['allocation_percentage'].last().reset_index()
    
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
    
    # Color palette for assets (matching the original charts)
    colors = {
        'SP500': '#87CEEB',  # Light blue
        'Gold': '#FFA500',   # Orange/Gold
        'US_REITs': '#90EE90',  # Light green
        'US_LT_Treas': '#4169E1',  # Royal blue
        'Intl_Dev_Stocks': '#9370DB',  # Medium purple
        'EM_Stocks': '#696969',  # Dim gray
        'Commodities': '#2F4F4F',  # Dark slate gray
        'Cash': '#D3D3D3'  # Light gray
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
        title_text="GAM Portfolio Performance Analysis"
    )
    
    return fig

def create_benchmark_performance_chart(data):
    """Create Benchmark Performance comparison with blue theme and professional styling"""
    df = data['benchmark_performance'].copy()
    
    # Blue theme palette for grouped bars (dark to light)
    blue_palette = ["#0B3C7D", "#1D5BBF", "#3C82F6", "#93B4FF"]  # YTD, 1Y, 5Y, SI
    single_colors = {
        'volatility': "#EF4444",  # Red
        'sharpe': "#22C55E",      # Green
        'beta': "#8B5CF6"         # Purple
    }
    
    # Create 2x2 subplots with proper spacing
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Returns Comparison (%)', 'Risk Metrics', 'Risk-Adjusted Returns', 'Beta to S&P 500'),
        horizontal_spacing=0.12,
        vertical_spacing=0.18
    )
    
    portfolios = df['portfolio'].tolist()
    
    # Returns comparison (grouped bars) - only these show in legend
    metrics_data = [
        ('ytd', 'YTD', blue_palette[0]),
        ('one_year', '1Y', blue_palette[1]),
        ('five_year', '5Y', blue_palette[2]),
        ('since_inception', 'Since Inception', blue_palette[3])
    ]
    
    for i, (metric, label, color) in enumerate(metrics_data):
        values = df[metric] * 100
        fig.add_trace(
            go.Bar(
                x=portfolios,
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
    
    # Risk metrics (Standard Deviation) - single series
    risk_values = df['standard_deviation'] * 100
    fig.add_trace(
        go.Bar(
            x=portfolios,
            y=risk_values,
            name='Volatility',
            marker_color=single_colors['volatility'],
            text=[f'{v:.1f}%' for v in risk_values],
            textposition='outside',
            textfont=dict(color=single_colors['volatility'], size=11),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Volatility: %{y:.2f}%<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Sharpe Ratio - single series
    sharpe_values = df['sharpe_ratio']
    fig.add_trace(
        go.Bar(
            x=portfolios,
            y=sharpe_values,
            name='Sharpe',
            marker_color=single_colors['sharpe'],
            text=[f'{v:.2f}' for v in sharpe_values],
            textposition='outside',
            textfont=dict(color=single_colors['sharpe'], size=11),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Sharpe Ratio: %{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Beta to S&P 500 - single series
    beta_values = df['beta_to_sp500']
    fig.add_trace(
        go.Bar(
            x=portfolios,
            y=beta_values,
            name='Beta',
            marker_color=single_colors['beta'],
            text=[f'{v:.2f}' for v in beta_values],
            textposition='outside',
            textfont=dict(color=single_colors['beta'], size=11),
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
        
        # Uniform text settings to prevent overlap
        uniformtext_minsize=10,
        uniformtext_mode="hide"
    )
    
    # Update y-axes with fixed ranges, titles, and styling
    fig.update_yaxes(
        range=[0, 16], 
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
    
    # Filter for trailing 12 months
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    twelve_months_ago = latest_date - timedelta(days=365)
    df = df[df['date'] >= twelve_months_ago]
    
    # Create month-year labels for x-axis - use month-end grouping to avoid duplicates
    df['month_end'] = df['date'].dt.to_period('M').dt.end_time
    df['month_year'] = df['month_end'].dt.strftime('%b')
    
    # Pivot data for stacked bar chart - use month_end for proper grouping
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
        'SP500': '#87CEEB',  # Light blue
        'Gold': '#FFA500',   # Orange/Gold
        'US_REITs': '#90EE90',  # Light green
        'US_LT_Treas': '#4169E1',  # Royal blue
        'Intl_Dev_Stocks': '#9370DB',  # Medium purple
        'EM_Stocks': '#696969',  # Dim gray
        'Commodities': '#2F4F4F',  # Dark slate gray
        'Cash': '#D3D3D3'  # Light gray
    }
    
    # Order columns same as allocation chart
    column_order = ['Commodities', 'EM_Stocks', 'Gold', 'Intl_Dev_Stocks', 'SP500', 'US_LT_Treas', 'US_REITs', 'Cash']
    available_columns = [col for col in column_order if col in pivot_df.columns]
    other_columns = [col for col in pivot_df.columns if col not in column_order]
    final_columns = available_columns + other_columns
    
    # Create ordered month list
    month_order = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    ordered_months = [month for month in month_order if month in pivot_df.index]
    pivot_df = pivot_df.reindex(ordered_months)
    
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

def calculate_ga_performance_metrics(monthly_returns_df):
    """Calculate GA performance metrics from monthly returns data"""
    df = monthly_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Current year and date calculations
    current_date = df['date'].max()
    current_year = current_date.year
    
    # YTD return
    ytd_data = df[df['date'].dt.year == current_year]
    ytd_return = (1 + ytd_data['gam_returns_net']).prod() - 1
    
    # 1-year return (trailing 12 months)
    one_year_ago = current_date - timedelta(days=365)
    one_year_data = df[df['date'] >= one_year_ago]
    one_year_return = (1 + one_year_data['gam_returns_net']).prod() - 1
    
    # 3-year return (annualized) - using exactly 36 months
    three_years_ago = current_date - timedelta(days=3*365.25)  # More precise calculation
    three_year_data = df[df['date'] >= three_years_ago]
    if len(three_year_data) >= 24:  # Ensure we have at least 2 years of data
        three_year_total_return = (1 + three_year_data['gam_returns_net']).prod() - 1
        years = len(three_year_data) / 12
        three_year_annualized = (1 + three_year_total_return) ** (1/years) - 1 if years > 0 else 0
    else:
        three_year_annualized = 0
    
    # 5-year return (annualized)
    five_years_ago = current_date - timedelta(days=5*365)
    five_year_data = df[df['date'] >= five_years_ago]
    if len(five_year_data) > 0:
        five_year_total_return = (1 + five_year_data['gam_returns_net']).prod() - 1
        years = len(five_year_data) / 12
        five_year_annualized = (1 + five_year_total_return) ** (1/years) - 1 if years > 0 else 0
    else:
        five_year_annualized = 0
    
    # Since inception return (annualized)
    inception_total_return = (1 + df['gam_returns_net']).prod() - 1
    years_since_inception = len(df) / 12
    since_inception_annualized = (1 + inception_total_return) ** (1/years_since_inception) - 1 if years_since_inception > 0 else 0
    
    # Risk metrics
    annual_volatility = df['gam_returns_net'].std() * np.sqrt(12)
    sharpe_ratio = (since_inception_annualized / annual_volatility) if annual_volatility > 0 else 0
    
    # Get beta from database instead of hardcoding
    conn = sqlite3.connect('portfolio_data.db')
    benchmark_data = pd.read_sql_query("SELECT beta_to_sp500 FROM benchmark_performance WHERE portfolio = 'GAM'", conn)
    beta_to_sp500 = benchmark_data['beta_to_sp500'].iloc[0] if not benchmark_data.empty else 0.75
    conn.close()
    
    # Max drawdown calculation
    cumulative_returns = (1 + df['gam_returns_net']).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min())
    
    # Downside deviation
    negative_returns = df['gam_returns_net'][df['gam_returns_net'] < 0]
    downside_deviation = negative_returns.std() * np.sqrt(12) if len(negative_returns) > 0 else 0
    
    return {
        'ytd': ytd_return,
        'one_year': one_year_return,
        'three_year': three_year_annualized,
        'five_year': five_year_annualized,
        'since_inception': since_inception_annualized,
        'standard_deviation': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'beta_to_sp500': beta_to_sp500,
        'max_drawdown': max_drawdown,
        'downside_deviation': downside_deviation
    }

def create_allocation_pie_chart(data):
    """Create current allocation pie chart"""
    df = data['gam_allocations'].copy()  # Use current GAM allocations
    
    # Get latest allocation
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    latest_allocation = df[df['date'] == latest_date]
    
    # Create asset name mapping for cleaner display
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
    
    # Apply name mapping
    latest_allocation['display_name'] = latest_allocation['asset_name'].map(asset_name_map).fillna(latest_allocation['asset_name'])
    
    fig = px.pie(
        latest_allocation,
        values='allocation_percentage',
        names='display_name',
        title=f'Current Asset Allocation (as of {latest_date.strftime("%B %Y")})',
        color_discrete_sequence=px.colors.qualitative.Set3
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
    ga_metrics = calculate_ga_performance_metrics(data['monthly_returns'])
    
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

def create_fact_sheet_landing(data, ga_metrics):
    # Page title
    st.markdown("<h1 style='text-align: center; color: #0f1419; margin-bottom: 2rem;'>RQA Global Adaptive Strategy Fact Sheet</h1>", unsafe_allow_html=True)
    
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
        
        performance_data = {
            '': ['RQA Global Adaptive', 'Global 60/40'],
            'YTD': [f"{ga_metrics['ytd']*100:.1f}%", 
                   f"{global_6040['ytd']*100:.1f}%" if global_6040 is not None else "N/A"],
            '1 Year': [f"{ga_metrics['one_year']*100:.1f}%", 
                      f"{global_6040['one_year']*100:.1f}%" if global_6040 is not None else "N/A"],
            '3 Year': [f"{ga_metrics['three_year']*100:.1f}%", 
                      f"{global_6040['five_year']*100:.1f}%" if global_6040 is not None else "N/A"],  # Using 5-year as proxy for 3-year
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
        df['date'] = pd.to_datetime(df['date'])
        latest_date = df['date'].max()
        latest_allocation = df[df['date'] == latest_date][['asset_name', 'allocation_percentage']].copy()
        latest_allocation = latest_allocation.sort_values('allocation_percentage', ascending=False)
        
        # Create better allocation display with proper formatting and asset renaming
        asset_name_map = {
            'SP500': 'S&P 500',
            'US_REITs': 'U.S. REITs', 
            'US_LT_Treas': 'U.S. LT Treas',
            'Intl_Dev_Stocks': 'Intl. Dev. Stocks',
            'EM_Stocks': 'EM Stocks',
            'Commodities': 'Commodities',
            'Gold': 'Gold',
            'Cash': 'Cash',
            # Full ETF name mappings
            'SPDR Portfolio S&P 500 ETF': 'S&P 500',
            'iShares Gold Trust': 'Gold',
            'Vanguard FTSE Developed Markets ETF': 'Intl. Dev. Stocks',
            'Vanguard Real Estate Index Fund': 'U.S. REITs',
            'Schwab Emerging Markets Equity ETF': 'EM Stocks',
            'iShares GSCI Commodity Dynamic Roll Strategy ETF': 'Commodities',
            'iShares 20+ Year Treasury Bond ETF': 'U.S. LT Treas',
            # Additional possible variants
            'iShares Core MSCI Total International Stock ETF': 'Intl. Dev. Stocks',
            'Vanguard Real Estate Index Fund ETF Shares': 'U.S. REITs',
            'Schwab Fundamental Emerging Markets Large Company Index Fund': 'EM Stocks',
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
            
            allocation = row['allocation_percentage']
            
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
    # Centered page title with better styling
    st.markdown("""
    <h1 style="text-align: center; 
               color: #0f1419; 
               margin-bottom: 2rem;
               font-size: 2.5rem;
               font-weight: bold;">RQA Portfolio Analytics Dashboard</h1>
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
            st.plotly_chart(create_allocation_pie_chart(data), width='stretch')
        
        with col2:
            # Show allocation table using current GAM data with abbreviated names
            df = data['gam_allocations'].copy()
            df['date'] = pd.to_datetime(df['date'])
            latest_date = df['date'].max()
            latest_allocation = df[df['date'] == latest_date][['asset_name', 'allocation_percentage']].copy()
            
            # Create asset name mapping for cleaner display
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
            
            # Apply name mapping
            latest_allocation['display_name'] = latest_allocation['asset_name'].map(asset_name_map).fillna(latest_allocation['asset_name'])
            
            # Create final dataframe with abbreviated names
            display_allocation = latest_allocation[['display_name', 'allocation_percentage']].copy()
            display_allocation = display_allocation.sort_values('allocation_percentage', ascending=False)
            display_allocation.columns = ['Asset Class', 'Allocation (%)']
            display_allocation['Allocation (%)'] = display_allocation['Allocation (%)'].round(2)
            
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
        st.plotly_chart(create_trailing_12m_allocations_chart(data), width='stretch')
        
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
        st.plotly_chart(create_monthly_returns_chart(data), width='stretch')
        
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
        st.plotly_chart(create_benchmark_performance_chart(data), width='stretch')
        
        
    
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
        st.plotly_chart(create_attribution_chart(data), width='stretch')
        
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
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Source:** GAM Portfolio Analytics Database | **Last Updated:** September 2025")

if __name__ == "__main__":
    main()
