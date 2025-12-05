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

try:
    from sqlalchemy import create_engine, text
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

st.set_page_config(
    page_title="Strategy Fact Sheet",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure default font for all plots
import plotly.io as pio
pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.font.family = "Merriweather, serif"

@st.cache_data(ttl=3600)
def load_data(strategy='GA'):
    """Load all data from the database (PostgreSQL primary, SQLite local fallback)
    
    Args:
        strategy: 'GA' or 'GB' to load specific strategy data
    """
    
    db_url = get_db_connection()
    
    if db_url and POSTGRES_AVAILABLE:
        cache_key = f"postgres_{hash(db_url) % 10000}_{strategy}"
    else:
        cache_key = f"sqlite_local_{strategy}"
    
    if 'last_db_type' not in st.session_state:
        st.session_state.last_db_type = cache_key
    elif st.session_state.last_db_type != cache_key:
        st.cache_data.clear()
        st.session_state.last_db_type = cache_key
    
    if db_url and POSTGRES_AVAILABLE:
        try:
            return load_data_postgres(strategy)
        except Exception as e:
            st.error(f"❌ PostgreSQL connection failed: {str(e)}")
            st.error("Please check your database connection in Streamlit secrets.")
            raise e
    elif POSTGRES_AVAILABLE:
        try:
            return load_data_postgres(strategy)
        except Exception as e:
            st.warning(f"PostgreSQL unavailable: {str(e)}. Using local SQLite.")
            return load_data_sqlite(strategy)
    else:
        return load_data_sqlite(strategy)

def get_db_connection():
    """Get database connection string from secrets"""
    try:
        if hasattr(st, 'secrets') and 'db_url' in st.secrets:
            return st.secrets['db_url']
        
        try:
            import toml
            if os.path.exists('secrets.toml'):
                secrets = toml.load('secrets.toml')
                db_url = secrets.get('database', {}).get('db_url')
                if db_url:
                    return db_url
        except ImportError:
            pass
        
        return None
    except Exception as e:
        # Suppress error display for missing secrets (expected in development)
        return None

def load_data_postgres(strategy='GA'):
    """Load data from PostgreSQL database
    
    Args:
        strategy: 'GA' or 'GB' to load specific strategy data
    """
    db_url = get_db_connection()
    if not db_url:
        raise Exception("No PostgreSQL connection string found in secrets")
    
    # Determine table names based on strategy
    if strategy == 'GB':
        returns_table = 'gbe_monthly_returns'
        allocations_table = 'gbe_allocations'
        attribution_table = 'gbe_attribution'
        returns_col = 'gbe_returns_net'
    else:  # GA
        returns_table = 'monthly_returns'
        allocations_table = 'ga_allocations'
        attribution_table = 'ga_attribution'
        returns_col = 'ga_returns_net'
    
    try:
        engine = create_engine(db_url, connect_args={"sslmode": "require"})
        
        with engine.connect() as test_conn:
            test_conn.execute(text("SELECT 1"))
        
        monthly_returns = pd.read_sql_query(f"SELECT * FROM {returns_table} ORDER BY date", engine)
        
        current_allocations = pd.read_sql_query(f"""
            SELECT * FROM {allocations_table} 
            WHERE date = (SELECT MAX(date) FROM {allocations_table})
            ORDER BY asset_symbol
        """, engine)
        
        historical_allocations = pd.read_sql_query(f"SELECT * FROM {allocations_table} ORDER BY date, asset_symbol", engine)
        historical_attribution = pd.read_sql_query(f"SELECT * FROM {attribution_table} ORDER BY date, asset_symbol", engine)
        
        # For benchmark, always use GA benchmark (it's the same for both strategies)
        benchmark_performance = pd.read_sql_query("SELECT * FROM benchmark_performance", engine)
        
        trailing_12m_allocations = historical_allocations.copy()
        trailing_12m_attribution = historical_attribution.copy()
        
        for df in [monthly_returns, current_allocations, historical_attribution, trailing_12m_allocations, trailing_12m_attribution]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        return {
            'monthly_returns': monthly_returns,
            'allocations': current_allocations,
            'attribution': historical_attribution,
            'benchmark_performance': benchmark_performance,
            'trailing_12m_allocations': trailing_12m_allocations,
            'trailing_12m_attribution': trailing_12m_attribution,
            'strategy': strategy,
            'returns_column': returns_col
        }
        
    except Exception as e:
        st.error(f"Database query failed: {str(e)}")
        raise e
    finally:
        if 'engine' in locals():
            engine.dispose()

def load_data_sqlite(strategy='GA'):
    """Load data from SQLite database (fallback)
    
    Args:
        strategy: 'GA' or 'GB' to load specific strategy data
    """
    if not os.path.exists('portfolio_data.db'):
        st.error("Database not found! Please run the migration script or ensure your database is available.")
        return create_empty_data_structure()
    
    # Determine table names based on strategy
    if strategy == 'GB':
        returns_table = 'gbe_monthly_returns'
        allocations_table = 'gbe_allocations'
        attribution_table = 'gbe_attribution'
        returns_col = 'gbe_returns_net'
    else:  # GA
        returns_table = 'monthly_returns'
        allocations_table = 'ga_allocations'
        attribution_table = 'ga_attribution'
        returns_col = 'ga_returns_net'
    
    conn = sqlite3.connect('portfolio_data.db')
    
    try:
        monthly_returns = pd.read_sql_query(f"SELECT * FROM {returns_table} ORDER BY date", conn)
        
        current_allocations = pd.read_sql_query(f"""
            SELECT * FROM {allocations_table} 
            WHERE date = (SELECT MAX(date) FROM {allocations_table})
            ORDER BY asset_symbol
        """, conn)
        
        historical_allocations = pd.read_sql_query(f"SELECT * FROM {allocations_table} ORDER BY date, asset_symbol", conn)
        historical_attribution = pd.read_sql_query(f"SELECT * FROM {attribution_table} ORDER BY date, asset_symbol", conn)
        
        # For benchmark, always use GA benchmark (it's the same for both strategies)
        benchmark_performance = pd.read_sql_query("SELECT * FROM benchmark_performance", conn)
        
        trailing_12m_allocations = historical_allocations.copy()
        trailing_12m_attribution = historical_attribution.copy()
        
        for df in [monthly_returns, current_allocations, historical_attribution, trailing_12m_allocations, trailing_12m_attribution]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        return {
            'monthly_returns': monthly_returns,
            'allocations': current_allocations,
            'attribution': historical_attribution,
            'benchmark_performance': benchmark_performance,
            'trailing_12m_allocations': trailing_12m_allocations,
            'trailing_12m_attribution': trailing_12m_attribution,
            'strategy': strategy,
            'returns_column': returns_col
        }
        
    finally:
        conn.close()

def create_empty_data_structure():
    """Create empty data structure for when database is not available"""
    empty_df = pd.DataFrame()
    return {
        'monthly_returns': empty_df,
        'allocations': empty_df,
        'attribution': empty_df,
        'benchmark_performance': empty_df,
        'trailing_12m_allocations': empty_df,
        'trailing_12m_attribution': empty_df,
        'strategy': 'GA',
        'returns_column': 'ga_returns_net'
    }

def create_growth_chart(data):
    """Create Growth of $100,000 chart showing portfolio vs benchmarks"""
    df = data['monthly_returns'].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    strategy = data.get('strategy', 'GA')
    returns_col = data.get('returns_column', 'ga_returns_net')
    strategy_name = 'RQA Global Adaptive' if strategy == 'GA' else 'RQA Global Balanced'
    
    # Filter start date for GB only
    # GA: starts from inception (2019) - no filtering
    # GB: starts from 2021-01-31 for consistent comparison
    if strategy == 'GB':
        start_date = pd.to_datetime('2021-01-31')
        df = df[df['date'] >= start_date].copy()
    
    initial_investment = 100000
    
    # Use hardcoded column names to match original GA implementation
    if strategy == 'GA':
        df['ga_cumulative'] = initial_investment * (1 + df['ga_returns_net']).cumprod()
        strategy_col = 'ga_cumulative'
        benchmark_col = 'portfolio_60_40'
        benchmark_name = 'Global 60/40'
    else:
        df['strategy_cumulative'] = initial_investment * (1 + df[returns_col]).cumprod()
        strategy_col = 'strategy_cumulative'
        benchmark_col = 'agg'
        benchmark_name = 'AGG'
    
    df['benchmark'] = initial_investment * (1 + df[benchmark_col]).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[strategy_col],
        mode='lines',
        name=strategy_name,
        line=dict(color='#1e3a8a', width=3),  # Dark blue
        hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['benchmark'],
        mode='lines',
        name=benchmark_name,
        line=dict(color='#6b7280', width=2),  # Grey
        hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))   
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
    df = data['trailing_12m_allocations'].copy()
    
    if df.empty:
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
    
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    
    if pd.isna(latest_date):
        pass
    else:
        latest_period = pd.Period(latest_date, freq='M')
        
        periods_to_include = [latest_period - i for i in range(11, -1, -1)] 
        
        df['period'] = df['date'].dt.to_period('M')
        df = df[df['period'].isin(periods_to_include)]
    
    monthly_df = df.groupby(['period', 'asset_name'])['allocation_percentage'].last().reset_index()
    
    monthly_df['month_end'] = monthly_df['period'].dt.end_time
    monthly_df = monthly_df.sort_values('period')
    
    pivot_df = monthly_df.pivot(index='month_end', columns='asset_name', values='allocation_percentage')
    
    pivot_df = pivot_df.fillna(0)
    
    pivot_df = pivot_df * 100
    
    pivot_df.index = pivot_df.index.strftime('%b %Y')
    
    fig = go.Figure()
    
    # Different color schemes based on strategy
    if data.get('strategy') == 'GB':
        # Structured color palette by asset class for GB
        colors = {
            # U.S. Equities (Green family - growth theme)
            'SP500': '#0B7A3E',
            
            # International Developed Equities (Purple family)
            'Europe_Stocks': '#5B3FA8',
            'Japan_Stocks': '#7A5BCC',
            'Intl_Dev_Stocks': '#A48AE8',
            
            # Emerging Markets (Warm reds)
            'EM_Stocks': '#B43D3D',
            
            # Real Estate REITs (Orange family)
            'US_REITs': '#C76B2E',
            'Intl_REITs': '#E19248',
            
            # Government Bonds (Blue family)
            'US_LT_Treas': '#133F73',
            'Intermediate_Bonds': '#5A8FD8',
            'TIPS': '#1D5F8A',
            
            # Corporate/Aggregate Bonds (Teal family)
            'Agg_Bonds': '#1F6F6E',
            'Intl_Bonds': '#79D0C1',
            
            # Commodities & Precious Metals (Gold/brown tones)
            'Commodities': '#A47C1B',
            'Gold': '#E3C448',
            
            # Cash (Gray - neutral)
            'Cash': '#4F4F4F',
            'IAGG': '#2A8F8C'
        }
    else:
        # Original blue palette for GA
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
    
    column_order = ['Commodities', 'EM_Stocks', 'Gold', 'Intl_Dev_Stocks', 'SP500', 'US_LT_Treas', 'US_REITs', 'Cash']
    available_columns = [col for col in column_order if col in pivot_df.columns]
    other_columns = [col for col in pivot_df.columns if col not in column_order]
    final_columns = available_columns + other_columns
    
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
        'Gold': 'Gold',
        # GB-specific assets
        'Agg_Bonds': 'Agg Bonds',
        'Europe_Stocks': 'Europe Stocks',
        'Intermediate_Bonds': 'Intermediate Bonds',
        'Intl_Bonds': 'Intl Bonds',
        'Intl_REITs': 'Intl REITs',
        'Japan_Stocks': 'Japan Stocks',
        'TIPS': 'TIPS'
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
        barmode='stack', 
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(range=[0, 100]),  
        xaxis=dict(tickangle=45)  
    )
    
    return fig

def create_monthly_returns_chart(data):
    """Create Monthly Returns chart with rolling volatility"""
    df = data['monthly_returns'].copy()
    returns_col = data.get('returns_column', 'ga_returns_net')
    strategy = data.get('strategy', 'GA')
    
    df['rolling_volatility'] = df[returns_col].rolling(window=12).std() * np.sqrt(12)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Returns (%)', 'Rolling 12-Month Volatility (%)'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    colors = ['green' if x >= 0 else 'red' for x in df[returns_col]]
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df[returns_col] * 100,
            name='Monthly Returns',
            marker_color=colors,
            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
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
    
    strategy_name = strategy if strategy == 'GA' else 'GB'
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text=f"{strategy_name} Portfolio Performance Analysis"
    )
    
    return fig

def create_benchmark_performance_chart(data, strategy='GA'):
    """Create Benchmark Performance comparison with blue theme and professional styling"""
    df = data['benchmark_performance'].copy()
    
    # Filter to show only strategy and its primary benchmark
    if strategy == 'GA':
        df = df[df['portfolio'].isin(['GA', '60/40'])]
    elif strategy == 'GB':
        df = df[df['portfolio'].isin(['GBE', 'AGG'])]  # Database uses GBE, not GB
    
    portfolio_colors = {
        'GA': "#1e3a8a",       # Dark blue for GA
        'GBE': "#7c3aed",      # Purple for GBE
        '60/40': "#15803d",    # Dark green for 60/40
        '70/30': "#6b7280",    # Gray for 70/30
        '50/50': "#ea580c",    # Orange for 50/50
        'AGG': "#0891b2"       # Cyan for AGG
    }
    
    time_period_colors = ["#0B3C7D", "#1D5BBF", "#3C82F6", "#93B4FF"]  # YTD, 1Y, 5Y, SI
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Returns Comparison (%)', 'Risk Metrics', 'Risk-Adjusted Returns', 'Beta to S&P 500'),
        horizontal_spacing=0.12,
        vertical_spacing=0.18
    )
    
    portfolios = df['portfolio'].tolist()
    
    # Map display names: GBE -> GB for display
    portfolio_display_names = []
    for p in portfolios:
        if p == 'GBE':
            portfolio_display_names.append('GB')
        else:
            portfolio_display_names.append(p)
    
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
                showlegend=True,
                hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.2f}%<extra></extra>"
            ),
            row=1, col=1
        )
    
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
    
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        title_text="Portfolio Benchmark Performance Comparison",
        title_font=dict(size=18),
        
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.05,
            yanchor="bottom",
            title=""
        ),
        
        margin=dict(l=60, r=30, t=60, b=60),
        
        barmode="group",
        bargap=0.18,
        bargroupgap=0.12,
        
        # Uniform text settings - allow text to show
        uniformtext_minsize=8,
        uniformtext_mode="show"
    )
    
    # Calculate dynamic ranges with padding for text labels
    # Returns - handle both positive and negative values
    returns_min = min((df['ytd'] * 100).min(), (df['one_year'] * 100).min(), 
                      (df['five_year'] * 100).min(), (df['since_inception'] * 100).min())
    returns_max = max((df['ytd'] * 100).max(), (df['one_year'] * 100).max(), 
                      (df['five_year'] * 100).max(), (df['since_inception'] * 100).max())
    returns_padding = (returns_max - returns_min) * 0.15  # 15% padding
    returns_range = [returns_min - returns_padding if returns_min < 0 else 0, 
                     returns_max + returns_padding]
    
    # Volatility - always non-negative, ensure proper padding
    volatility_min = (df['standard_deviation'] * 100).min()
    volatility_max = (df['standard_deviation'] * 100).max()
    volatility_padding = (volatility_max - volatility_min) * 0.15
    volatility_range = [0, volatility_max + volatility_padding]
    
    # Sharpe Ratio - can be negative, handle accordingly
    sharpe_min = df['sharpe_ratio'].min()
    sharpe_max = df['sharpe_ratio'].max()
    sharpe_padding = (sharpe_max - sharpe_min) * 0.15
    sharpe_range = [sharpe_min - sharpe_padding if sharpe_min < 0 else 0, 
                    sharpe_max + sharpe_padding]
    
    # Beta - typically positive but can vary
    beta_min = df['beta_to_sp500'].min()
    beta_max = df['beta_to_sp500'].max()
    beta_padding = (beta_max - beta_min) * 0.15
    beta_range = [max(0, beta_min - beta_padding), beta_max + beta_padding]
    
    # Update y-axes with dynamic ranges, titles, and styling
    fig.update_yaxes(
        range=returns_range,
        title_text="Returns (%)",
        tickformat=".1f",
        gridcolor="rgba(15,23,42,0.08)",
        showgrid=True,
        zeroline=False,
        row=1, col=1
    )
    
    fig.update_yaxes(
        range=volatility_range, 
        title_text="Volatility (%)",
        tickformat=".1f",
        gridcolor="rgba(15,23,42,0.08)",
        showgrid=True,
        zeroline=False,
        row=1, col=2
    )
    
    fig.update_yaxes(
        range=sharpe_range, 
        title_text="Sharpe Ratio",
        tickformat=".2f",
        gridcolor="rgba(15,23,42,0.08)",
        showgrid=True,
        zeroline=False,
        row=2, col=1
    )
    
    fig.update_yaxes(
        range=beta_range, 
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
    
    fig.update_annotations(font=dict(size=14))
    
    return fig

def create_attribution_chart(data):
    """Create Portfolio Attribution stacked bar chart"""
    df = data['trailing_12m_attribution'].copy()
    
    if df.empty:
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
    
    # Attribution data is stored as decimals (e.g., 0.00281 = 0.281%)
    # So multiply by 100 to get percentage values for display
    pivot_df = pivot_df * 100
    
    fig = go.Figure()
    
    # Determine strategy from data context
    strategy = data.get('strategy', 'GA')
    
    if strategy == 'GB':
        # Structured color palette by asset class for GB (matching other charts)
        colors = {
            # U.S. Equities (Green)
            'SP500': '#0B7A3E',
            
            # International Developed Equities (Purple family)
            'Europe_Stocks': '#5B3FA8',
            'Japan_Stocks': '#7A5BCC',
            'Intl_Dev_Stocks': '#A48AE8',
            
            # Emerging Markets (Warm red)
            'EM_Stocks': '#B43D3D',
            
            # Real Estate REITs (Orange family)
            'US_REITs': '#C76B2E',
            'Intl_REITs': '#E19248',
            
            # Government Bonds (Blue family)
            'US_LT_Treas': '#133F73',
            'Intermediate_Bonds': '#5A8FD8',
            'TIPS': '#1D5F8A',
            
            # Corporate/Aggregate Bonds (Teal family)
            'Agg_Bonds': '#1F6F6E',
            'Intl_Bonds': '#79D0C1',
            
            # Commodities & Precious Metals (Gold/brown)
            'Commodities': '#A47C1B',
            'Gold': '#E3C448',
            
            # Cash (Gray)
            'Cash': '#4F4F4F',
            'IAGG': '#2A8F8C'
        }
    else:
        # Original blue palette for GA
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
        asset_str = str(asset) if pd.notna(asset) else 'Unknown'
        
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=pivot_df[asset],
            name=asset_str.replace('_', ' ').replace('SP500', 'S&P 500').replace('US LT Treas', 'U.S. LT Treas').replace('US REITs', 'U.S. REITs').replace('Intl Dev Stocks', 'Intl. Dev. Stocks').replace('EM Stocks', 'EM Stocks'),
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
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    
    return fig

def calculate_ga_performance_metrics(data_dict):
    """Get performance metrics from database and calculate 3-year return"""
    benchmark_data = data_dict['benchmark_performance']
    monthly_returns_df = data_dict['monthly_returns']
    strategy = data_dict.get('strategy', 'GA')
    returns_col = data_dict.get('returns_column', 'ga_returns_net')

    # For GA, use GA benchmark. For GB, calculate metrics from monthly returns
    if strategy == 'GA':
        ga_benchmark = benchmark_data[benchmark_data['portfolio'] == 'GA']
        
        if not ga_benchmark.empty:
            ga_data = ga_benchmark.iloc[0]
            
            # Calculate 3-year return from monthly returns data
            df = monthly_returns_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            current_date = df['date'].max()
            three_years_ago = current_date - timedelta(days=3*365.25)
            three_year_data = df[df['date'] >= three_years_ago]
            
            if len(three_year_data) >= 24:
                three_year_total_return = (1 + three_year_data[returns_col]).prod() - 1
                years = len(three_year_data) / 12
                three_year_annualized = (1 + three_year_total_return) ** (1/years) - 1 if years > 0 else 0
            else:
                three_year_annualized = 0.0
            
            return {
                'ytd': ga_data['ytd'],
                'one_year': ga_data['one_year'],
                'three_year': three_year_annualized,
                'five_year': ga_data['five_year'],
                'since_inception': ga_data['since_inception'],
                'standard_deviation': ga_data['standard_deviation'],
                'sharpe_ratio': ga_data['sharpe_ratio'],
                'beta_to_sp500': ga_data['beta_to_sp500'],
                'max_drawdown': 0.0,
                'downside_deviation': 0.0
            }
    
    # For GB, calculate all metrics from monthly returns
    df = monthly_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    if returns_col not in df.columns:
        return {
            'ytd': 0.0, 'one_year': 0.0, 'three_year': 0.0, 'five_year': 0.0,
            'since_inception': 0.0, 'standard_deviation': 0.0, 'sharpe_ratio': 0.0,
            'beta_to_sp500': 0.0, 'max_drawdown': 0.0, 'downside_deviation': 0.0
        }
    
    returns = df[returns_col].dropna()
    
    if len(returns) < 2:
        # Return what we can calculate with limited data
        if len(returns) == 1:
            return {
                'ytd': returns.iloc[0],
                'one_year': 0.0,
                'three_year': 0.0,
                'five_year': 0.0,
                'since_inception': returns.iloc[0],
                'standard_deviation': 0.0,
                'sharpe_ratio': 0.0,
                'beta_to_sp500': 0.0,
                'max_drawdown': 0.0,
                'downside_deviation': 0.0
            }
        else:
            return {
                'ytd': 0.0, 'one_year': 0.0, 'three_year': 0.0, 'five_year': 0.0,
                'since_inception': 0.0, 'standard_deviation': 0.0, 'sharpe_ratio': 0.0,
                'beta_to_sp500': 0.0, 'max_drawdown': 0.0, 'downside_deviation': 0.0
            }
    
    current_date = df['date'].max()
    
    # YTD
    current_year = current_date.year
    ytd_data = returns[df['date'].dt.year == current_year]
    ytd_return = (1 + ytd_data).prod() - 1 if len(ytd_data) > 0 else 0.0
    
    # 1-Year - need at least 11 months
    one_year_ago = current_date - timedelta(days=365.25)
    one_year_data = returns[df['date'] >= one_year_ago]
    if len(one_year_data) >= 11:
        one_year_total = (1 + one_year_data).prod() - 1
        years = len(one_year_data) / 12
        one_year_return = (1 + one_year_total) ** (1/years) - 1 if years > 0 else 0
    else:
        one_year_return = None  # Not enough data
    
    # 3-Year - need at least 30 months
    three_years_ago = current_date - timedelta(days=3*365.25)
    three_year_data = returns[df['date'] >= three_years_ago]
    if len(three_year_data) >= 30:
        three_year_total = (1 + three_year_data).prod() - 1
        years = len(three_year_data) / 12
        three_year_return = (1 + three_year_total) ** (1/years) - 1 if years > 0 else 0
    else:
        three_year_return = None  # Not enough data
    
    # 5-Year - need at least 48 months (4 years) to estimate 5-year annualized
    five_years_ago = current_date - timedelta(days=5*365.25)
    five_year_data = returns[df['date'] >= five_years_ago]
    if len(five_year_data) >= 48:
        five_year_total = (1 + five_year_data).prod() - 1
        years = len(five_year_data) / 12
        five_year_return = (1 + five_year_total) ** (1/years) - 1 if years > 0 else 0
    else:
        five_year_return = None  # Not enough data
    
    # Since Inception - need at least 2 months
    if len(returns) >= 2:
        cumulative_return = (1 + returns).prod() - 1
        years = len(returns) / 12
        since_inception = (1 + cumulative_return) ** (1/years) - 1 if years > 0 else 0
    else:
        since_inception = returns.iloc[0] if len(returns) == 1 else 0.0
    
    # Risk metrics - need at least 12 months for meaningful volatility
    if len(returns) >= 12:
        std_dev = returns.std() * np.sqrt(12)
    else:
        std_dev = None  # Not enough data for annualized vo latility
    
    # Sharpe ratio - need both return and volatility
    if std_dev is not None and std_dev > 0:
        annualized_return = since_inception
        sharpe = annualized_return / std_dev
    else:
        sharpe = None
    
    # Beta to S&P 500 - need at least 12 months
    if len(returns) >= 12:
        spy_returns = df['spy'].dropna()
        aligned_returns = returns[returns.index.isin(spy_returns.index)]
        aligned_spy = spy_returns[spy_returns.index.isin(returns.index)]
        
        if len(aligned_returns) >= 12 and len(aligned_spy) >= 12:
            covariance = np.cov(aligned_returns, aligned_spy)[0][1]
            spy_variance = np.var(aligned_spy)
            beta = covariance / spy_variance if spy_variance > 0 else None
        else:
            beta = None
    else:
        beta = None
    
    return {
        'ytd': ytd_return,
        'one_year': one_year_return,
        'three_year': three_year_return,
        'five_year': five_year_return,
        'since_inception': since_inception,
        'standard_deviation': std_dev,
        'sharpe_ratio': sharpe,
        'beta_to_sp500': beta,
        'max_drawdown': 0.0,
        'downside_deviation': 0.0
    }

def create_allocation_pie_chart(data):
    """Create current allocation pie chart"""
    df = data['allocations'].copy()  # Use current allocations
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No allocation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title='Target Asset Allocation',
            height=500
        )
        return fig
    
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    
    asset_name_map = {
        'SPDR Portfolio S&P 500 ETF': 'U.S. Stocks (S&P 500)',
        'iShares Gold Trust': 'Gold',
        'Vanguard FTSE Developed Markets ETF': 'International Developed Stocks',
        'Vanguard Real Estate Index Fund': 'U.S. Real Estate (REITs)',
        'Schwab Emerging Markets Equity ETF': 'Emerging Market Stocks',
        'iShares GSCI Commodity Dynamic Roll Strategy ETF': 'Commodities',
        'iShares 20+ Year Treasury Bond ETF': 'U.S. LT Treas',
        'Cash': 'Cash',
        'SP500': 'U.S. Stocks (S&P 500)',
        'US_REITs': 'U.S. Real Estate (REITs)', 
        'US_LT_Treas': 'U.S. LT Treas',
        'Intl_Dev_Stocks': 'International Developed Stocks',
        'EM_Stocks': 'Emerging Market Stocks',
        'Commodities': 'Commodities',
        'Gold': 'Gold',
        # GB-specific assets
        'Agg_Bonds': 'Agg Bonds',
        'Europe_Stocks': 'Europe Stocks',
        'Intermediate_Bonds': 'Intermediate Bonds',
        'Intl_Bonds': 'Intl Bonds',
        'Intl_REITs': 'Intl REITs',
        'Japan_Stocks': 'Japan Stocks',
        'TIPS': 'TIPS'
    }
    
    # Handle case where latest_date is NaT (no valid dates)
    if pd.isna(latest_date):
        date_str = "Latest Available"
        latest_allocation = df.iloc[-len(df.groupby('asset_name')):]  # Get last entry for each asset
    else:
        date_str = latest_date.strftime("%B %Y")
        latest_allocation = df[df['date'] == latest_date]
    
    latest_allocation['display_name'] = latest_allocation['asset_name'].map(asset_name_map).fillna(latest_allocation['asset_name'])
    
    # Combine allocations that map to the same display name (e.g., Cash + US_LT_Treas both become "Cash")
    latest_allocation = latest_allocation.groupby('display_name')['allocation_percentage'].sum().reset_index()
    
    # Filter out zero or near-zero allocations
    latest_allocation = latest_allocation[latest_allocation['allocation_percentage'] > 0.001]  # Greater than 0.1%
    
    # Check if we have any data after filtering
    if latest_allocation.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No allocation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title='Target Asset Allocation',
            height=500
        )
        return fig
    
    # Determine strategy from data context
    strategy = data.get('strategy', 'GA')
    
    if strategy == 'GB':
        # Structured color palette by asset class for GB (matching trailing 12m chart)
        colors = {
            # U.S. Equities (Green)
            'U.S. Stocks (S&P 500)': '#0B7A3E',
            'SP500': '#0B7A3E',
            
            # International Developed Equities (Purple family)
            'Europe Stocks': '#5B3FA8',
            'Japan Stocks': '#7A5BCC',
            'International Developed Stocks': '#A48AE8',
            'Intl_Dev_Stocks': '#A48AE8',
            
            # Emerging Markets (Warm red)
            'Emerging Market Stocks': '#B43D3D',
            'EM_Stocks': '#B43D3D',
            
            # Real Estate REITs (Orange family)
            'U.S. Real Estate (REITs)': '#C76B2E',
            'US_REITs': '#C76B2E',
            'Intl REITs': '#E19248',
            
            # Government Bonds (Blue family)
            'U.S. LT Treas': '#133F73',
            'Intermediate Bonds': '#5A8FD8',
            'TIPS': '#1D5F8A',
            
            # Corporate/Aggregate Bonds (Teal family)
            'Agg Bonds': '#1F6F6E',
            'Intl Bonds': '#79D0C1',
            
            # Commodities & Precious Metals (Gold/brown)
            'Commodities': '#A47C1B',
            'Gold': '#E3C448',
            
            # Cash (Gray)
            'Cash': '#4F4F4F'
        }
    else:
        # Original blue palette for GA
        colors = {
            'U.S. Stocks (S&P 500)': '#80c1ff',
            'SP500': '#80c1ff',
            'Gold': '#4da6ff',
            'U.S. Real Estate (REITs)': '#337fcc',
            'US_REITs': '#337fcc',
            'U.S. LT Treas': '#1a5fb4',
            'International Developed Stocks': '#00509e',
            'Intl_Dev_Stocks': '#00509e',
            'Emerging Market Stocks': '#004080',
            'EM_Stocks': '#004080',
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
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Strategy selector in sidebar
    st.sidebar.image("logo.png", width=200)
    st.sidebar.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    strategy = st.sidebar.selectbox(
        "Select Strategy",
        ["Global Adaptive (GA)", "Global Balanced (GB)"],
        index=0
    )
    
    # Parse strategy code
    strategy_code = 'GA' if 'GA' in strategy else 'GB'
    
    data = load_data(strategy_code)
    
    ga_metrics = calculate_ga_performance_metrics(data)
    
    # Main navigation choice
    view_option = st.sidebar.selectbox(
        "Navigation",
        ["Strategy Fact Sheet", "Interactive Analytics"],
        index=0
    )
    
    if view_option == "Strategy Fact Sheet":
        create_fact_sheet_landing(data, ga_metrics, strategy_code)
    else:
        create_analytics_dashboard(data, ga_metrics, strategy_code)



def create_fact_sheet_landing(data, ga_metrics, strategy='GA'):
    # Clean Professional Header with Logo and Strategy Name
    import base64
    import os
    
    strategy_title = "GLOBAL ADAPTIVE" if strategy == 'GA' else "GLOBAL BALANCED"
    
    logo_html = ""
    if os.path.exists("logo.png"):
        with open("logo.png", "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{logo_data}" style="width: auto; height: 100px; border-radius: 12px; background: white; padding: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);"/>'
        
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
        padding: 1.5rem 3rem;
        margin: -1rem -1rem 3rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        min-height: 90px;
        position: relative;
    ">
        <div style="flex-shrink: 0;">
            {logo_html}
        </div>
        <div style="flex: 1;"></div>
        <div style="text-align: right; flex-shrink: 0; margin-right: -1rem;">
            <h1 style="
                margin: 0;
                font-size: 2.5rem;
                font-weight: 300;
                letter-spacing: 0.2rem;
                color: white;
                line-height: 1.1;
                text-shadow: 0 3px 6px rgba(0,0,0,0.4);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            ">{strategy_title}</h1>
            <div style="
                width: 280px; 
                height: 2px; 
                background: white; 
                margin: 0.5rem 0 0.5rem auto;
            "></div>
            <h2 style="
                margin: 0;
                font-size: 1.6rem;
                font-weight: 300;
                letter-spacing: 0.3rem;
                color: white;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            ">STRATEGY</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)


    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: bold;">STRATEGY OVERVIEW</h2>
        </div>
        """, unsafe_allow_html=True)

        #To scale, we would replace the else with elif, and keep adding more strategies
        if strategy == 'GA':
            st.markdown("""
            Global Adaptive (GA) is designed to meet the challenges of today's ever-changing markets by allocating capital across major global asset classes through a disciplined, data-driven process. Traditional approaches, such as the 60/40 stock-bond portfolio, rely on static assumptions that may leave investors overexposed when conditions shift.
            
            Instead, GA takes a dynamic approach. By continuously evaluating global opportunities, the strategy seeks to increase exposure to asset classes showing strength while reducing or avoiding those losing momentum. This adaptability allows the portfolio to respond proactively rather than reactively, providing the potential for stronger risk-adjusted returns over time.
            
            Through this evidence-based framework, GA strives to deliver three key benefits: more consistent growth, enhanced stability, and improved capital preservation. The result is a portfolio designed not only to participate in market gains, but also to better withstand periods of stress — an advantage that can compound meaningfully for investors over the long run.
            """)
        else: 
            st.markdown("""
            Global Balanced (GB) is a diversified portfolio strategy that invests in major asset classes globally through liquid exchanged traded funds (ETFs). GB takes a dynamic, all-weather approach to portfolio construction in order to seek out more stable performance across economic environments. The strategy targets a conservative capital appreciation profile while minimizing portfolio volatility and risk through enhanced diversification and portfolio balancing techniques.
            """)
        
        st.plotly_chart(create_growth_chart(data), use_container_width=True)
        
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: bold;">STRATEGY RETURNS (Net of Fees)</h2>
        </div>
        """, unsafe_allow_html=True)
        
        benchmark_data = data['benchmark_performance'].copy()
        
        strategy_display_name = 'RQA Global Adaptive' if strategy == 'GA' else 'RQA Global Balanced'
        
        # Select benchmark portfolio based on strategy
        if strategy == 'GA':
            benchmark_portfolio = '60/40'
            benchmark_display_name = 'Global 60/40'
            benchmark_col = 'portfolio_60_40'
        elif strategy == 'GB':  
            benchmark_portfolio = 'AGG'
            benchmark_display_name = 'AGG'
            benchmark_col = 'agg'
        
        benchmark_perf = benchmark_data[benchmark_data['portfolio'] == benchmark_portfolio].iloc[0] if not benchmark_data.empty else None
        
        # Calculate 3-year and 5-year returns for benchmarks from monthly returns data
        monthly_returns_df = data['monthly_returns']
        df = monthly_returns_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        current_date = df['date'].max()
        three_years_ago = current_date - timedelta(days=3*365.25)
        five_years_ago = current_date - timedelta(days=5*365.25)
        three_year_data = df[df['date'] >= three_years_ago]
        five_year_data = df[df['date'] >= five_years_ago]
        
        # Calculate 3-year for main benchmark
        if len(three_year_data) >= 24:
            three_year_total_return_benchmark = (1 + three_year_data[benchmark_col]).prod() - 1
            years = len(three_year_data) / 12
            three_year_annualized_benchmark = (1 + three_year_total_return_benchmark) ** (1/years) - 1 if years > 0 else 0
        else:
            three_year_annualized_benchmark = 0.0
        
        # Calculate 5-year for main benchmark
        if len(five_year_data) >= 48:
            five_year_total_return_benchmark = (1 + five_year_data[benchmark_col]).prod() - 1
            years = len(five_year_data) / 12
            five_year_annualized_benchmark = (1 + five_year_total_return_benchmark) ** (1/years) - 1 if years > 0 else None
        else:
            five_year_annualized_benchmark = None
        
        performance_data = {
            '': [strategy_display_name, benchmark_display_name],
            'YTD': [f"{ga_metrics['ytd']*100:.1f}%" if ga_metrics['ytd'] is not None else "N/A", 
                   f"{benchmark_perf['ytd']*100:.1f}%" if benchmark_perf is not None else "N/A"],
            '1 Year': [f"{ga_metrics['one_year']*100:.1f}%" if ga_metrics['one_year'] is not None else "N/A", 
                      f"{benchmark_perf['one_year']*100:.1f}%" if benchmark_perf is not None else "N/A"],
            '3 Year': [f"{ga_metrics['three_year']*100:.1f}%" if ga_metrics['three_year'] is not None else "N/A", 
                      f"{three_year_annualized_benchmark*100:.1f}%"],
            '5 Year': [f"{ga_metrics['five_year']*100:.1f}%" if ga_metrics['five_year'] is not None else "N/A", 
                      f"{five_year_annualized_benchmark*100:.1f}%" if five_year_annualized_benchmark is not None else "N/A"],
            'Since Inception': [f"{ga_metrics['since_inception']*100:.1f}%" if ga_metrics['since_inception'] is not None else "N/A", 
                              f"{benchmark_perf['since_inception']*100:.1f}%" if benchmark_perf is not None else "N/A"]
        }
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, hide_index=True, use_container_width=True)
        
        # Risk Management with consistent header format and table
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: bold;">RISK MANAGEMENT</h2>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1rem; font-style: italic; font-weight: 300;">"Risk Forward Framework"</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Format metrics safely, handling None values
        def fmt_pct(val):
            return f"{val*100:.1f}%" if val is not None else "N/A"
        
        def fmt_ratio(val):
            return f"{val:.2f}" if val is not None else "N/A"
        
        # Both GA and GB now have 2 columns (strategy + benchmark)
        risk_data = {
            'Metric': ['Annualized Volatility', 'Sharpe Ratio (RF=0)', 'Beta (vs. S&P)'],
            strategy: [fmt_pct(ga_metrics.get('standard_deviation')), 
                       fmt_ratio(ga_metrics.get('sharpe_ratio')), 
                       fmt_ratio(ga_metrics.get('beta_to_sp500'))],
            benchmark_display_name: [fmt_pct(benchmark_perf['standard_deviation']) if benchmark_perf is not None else "N/A",
                            fmt_ratio(benchmark_perf['sharpe_ratio']) if benchmark_perf is not None else "N/A",
                            fmt_ratio(benchmark_perf['beta_to_sp500']) if benchmark_perf is not None else "N/A"]
        }
        
        df_risk = pd.DataFrame(risk_data)
        st.dataframe(df_risk, hide_index=True, use_container_width=True)
        
        # Key Strategy Features (move above Portfolio Management)
        st.markdown("""
        <div style="background: #3b5998; color: white; padding: 0.9rem 1.4rem; border-radius: 8px; margin: 1rem 0 0.4rem 0;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: bold;">KEY STRATEGY FEATURES</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if strategy == 'GA':
            st.markdown("""
            <ul style="list-style:none; padding:0; margin:0 0 1rem 0;">
                <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem;">
                    <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                    <span style="line-height:1.15;">Tactical Asset Allocation</span>
                </li>
                <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem;">
                    <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                    <span style="line-height:1.15;">Global Diversification</span>
                </li>
                <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem;">
                    <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                    <span style="line-height:1.15;">Systematic Risk Management</span>
                </li>
                <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem;">
                    <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                    <span style="line-height:1.15;">Daily Liquidity</span>
                </li>
                <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0; font-size:0.95rem;">
                    <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                    <span style="line-height:1.15;">Monthly Rebalancing</span>
                </li>
            </ul>
            """, unsafe_allow_html=True)
        else:  # GB
            st.markdown("""
            <ul style="list-style:none; padding:0; margin:0 0 1rem 0;">
                <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem;">
                    <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                    <span style="line-height:1.15;">Global Asset Allocation</span>
                </li>
                <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem;">
                    <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                    <span style="line-height:1.15;">Intelligently Balanced</span>
                </li>
                <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem;">
                    <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                    <span style="line-height:1.15;">Evolves with the Market</span>
                </li>
                <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0 0 0.5rem 0; font-size:0.95rem;">
                    <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                    <span style="line-height:1.15;">"Quantamental"</span>
                </li>
                <li style="display:flex; align-items:flex-start; gap:0.55rem; margin:0; font-size:0.95rem;">
                    <span style="color:#3b5998; font-weight:600; line-height:1; padding-top:2px;">•</span>
                    <span style="line-height:1.15;">Diversification First</span>
                </li>
            </ul>
            """, unsafe_allow_html=True)
        
        # Portfolio Management with consistent header format
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: bold;">PORTFOLIO MANAGEMENT</h2>
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
        st.markdown("""
        <div style="background: #3b5998;
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: bold;">KEY FACTS</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if strategy == 'GA':
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
        else:  # GB
            st.markdown("""
            **INCEPTION DATE:**  
            January 2019
            
            **STYLE:**  
            Multi-Asset
            
            **BENCHMARK:**  
            AGG (U.S. Aggregate Bond)
            
            **ASSET UNIVERSE:**  
            U.S. Equities, Developed & Emerging Market Equities, Global Real Estate Investment Trusts (REIT's), Global Bonds, U.S. Treasuries, Precious Metals, and Global Commodity Composites
            
            **OFFERING VEHICLE:**  
            Separately Managed Accounts & Index Delivery
            
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
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold; line-height: 1.1;">TARGET<br>ALLOCATION</h2>
        </div>
        """, unsafe_allow_html=True)
        
        df = data['allocations'].copy()
        
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
                'US_LT_Treas': 'U.S. LT Treas',
                'Intl_Dev_Stocks': 'International Developed Stocks',
                'EM_Stocks': 'Emerging Market Stocks',
                'Commodities': 'Commodities',
                'Gold': 'Gold',
                'Cash': 'Cash',
                # GB-specific assets
                'Agg_Bonds': 'Agg Bonds',
                'Europe_Stocks': 'Europe Stocks',
                'Intermediate_Bonds': 'Intermediate Bonds',
                'Intl_Bonds': 'Intl Bonds',
                'Intl_REITs': 'Intl REITs',
                'Japan_Stocks': 'Japan Stocks',
                'TIPS': 'TIPS',
                # ETF full names
                'SPDR Portfolio S&P 500 ETF': 'U.S. Stocks (S&P 500)',
                'iShares Gold Trust': 'Gold',
                'Vanguard FTSE Developed Markets ETF': 'International Developed Stocks',
                'Vanguard Real Estate Index Fund': 'U.S. Real Estate (REITs)',
                'Schwab Emerging Markets Equity ETF': 'Emerging Market Stocks',
                'iShares GSCI Commodity Dynamic Roll Strategy ETF': 'Commodities',
                'iShares 20+ Year Treasury Bond ETF': 'U.S. LT Treas',
                'iShares Core MSCI Total International Stock ETF': 'International Developed Stocks',
                'Vanguard Real Estate Index Fund ETF Shares': 'U.S. Real Estate (REITs)',
                'Schwab Fundamental Emerging Markets Large Company Index Fund': 'Emerging Market Stocks',
                'Invesco DB Commodity Index Tracking Fund': 'Commodities'
            }
            
            latest_allocation['display_name'] = latest_allocation['asset_name'].map(asset_name_map).fillna(latest_allocation['asset_name'])
            
            # Combine allocations that map to the same display name (e.g., Cash + US_LT_Treas both become "Cash")
            combined_allocation = latest_allocation.groupby('display_name')['allocation_percentage'].sum().reset_index()
            combined_allocation = combined_allocation.sort_values('allocation_percentage', ascending=False)
            
            for _, row in combined_allocation.iterrows():
                display_name = row['display_name']
                
                allocation = row['allocation_percentage'] * 100  # Convert to percentage
                
                # Show more decimal places for small allocations (like cash)
                if allocation < 1.0:
                    allocation_str = f"{allocation:.2f}%"
                else:
                    allocation_str = f"{allocation:.1f}%"
                
                bar_width = int(allocation * 2)
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



    st.markdown("""
    <div style="margin-top: 2rem; 
                padding: 1rem; 
                background: linear-gradient(135deg,
                border-radius: 8px; 
                font-size: 0.85rem; 
                color:
                border-left: 4px solid #9ca3af;">
        <strong>Important Disclosures:</strong> This presentation is for informational purposes only. Past performance does not guarantee future results. 
        All investments involve risk of loss. The strategy may not be suitable for all investors. Please consult with a financial advisor before investing.
        Performance figures are presented net of estimated management fees and related ETF fees and expenses.
    </div>
    """, unsafe_allow_html=True)

def create_analytics_dashboard(data, ga_metrics, strategy='GA'):
    # Clean Professional Header with Logo and Analytics Name
    import base64
    import os
    
    logo_html = ""
    if os.path.exists("logo.png"):
        with open("logo.png", "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{logo_data}" style="width: auto; height: 100px; border-radius: 12px; background: white; padding: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);"/>'
    
    # Create the professional header with logo on left and analytics name on right
    
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
        padding: 1.5rem 3rem;
        margin: -1rem -1rem 3rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        min-height: 90px;
        position: relative;
    ">
        <div style="flex-shrink: 0;">
            {logo_html}
        </div>
        <div style="flex: 1;"></div>
        <div style="text-align: right; flex-shrink: 0; margin-right: -1rem;">
            <h1 style="
                margin: 0;
                font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                font-size: 2.5rem;
                font-weight: 300;
                letter-spacing: 0.2rem;
                color: white;
                line-height: 1.1;
                text-shadow: 0 3px 6px rgba(0,0,0,0.4);
            ">PORTFOLIO</h1>
            <div style="
                width: 280px; 
                height: 2px; 
                background: white; 
                margin: 0.5rem 0 0.5rem auto;
            "></div>
            <h2 style="
                margin: 0;
                font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                font-size: 1.6rem;
                font-weight: 300;
                letter-spacing: 0.3rem;
                color: white;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            ">ANALYTICS</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("Performance Metrics")
    st.sidebar.metric("YTD Return", f"{ga_metrics['ytd']*100:.2f}%" if ga_metrics['ytd'] is not None else "N/A")
    st.sidebar.metric("1-Year Return", f"{ga_metrics['one_year']*100:.2f}%" if ga_metrics['one_year'] is not None else "N/A")
    st.sidebar.metric("5-Year Return", f"{ga_metrics['five_year']*100:.2f}%" if ga_metrics['five_year'] is not None else "N/A")
    st.sidebar.metric("Since Inception", f"{ga_metrics['since_inception']*100:.2f}%" if ga_metrics['since_inception'] is not None else "N/A")
    st.sidebar.metric("Sharpe Ratio", f"{ga_metrics['sharpe_ratio']:.3f}" if ga_metrics['sharpe_ratio'] is not None else "N/A")
    st.sidebar.metric("Volatility", f"{ga_metrics['standard_deviation']*100:.2f}%" if ga_metrics['standard_deviation'] is not None else "N/A")
    st.sidebar.metric("Beta to S&P 500", f"{ga_metrics['beta_to_sp500']:.3f}" if ga_metrics['beta_to_sp500'] is not None else "N/A")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Target Allocation", 
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
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: bold;">TARGET PORTFOLIO ALLOCATION</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Add some spacing
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1.2, 0.8], gap="large")
        
        with col1:
            st.plotly_chart(create_allocation_pie_chart(data), use_container_width=True)
        
        with col2:
            st.markdown("### Target Allocation Breakdown")
            # Show allocation table using current data with abbreviated names
            df = data['allocations'].copy()
            
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
                
                asset_name_map = {
                    'SPDR Portfolio S&P 500 ETF': 'U.S. Stocks (S&P 500)',
                    'iShares Gold Trust': 'Gold',
                    'Vanguard FTSE Developed Markets ETF': 'International Developed Stocks',
                    'Vanguard Real Estate Index Fund': 'U.S. Real Estate (REITs)',
                    'Schwab Emerging Markets Equity ETF': 'Emerging Market Stocks',
                    'iShares GSCI Commodity Dynamic Roll Strategy ETF': 'Commodities',
                    'iShares 20+ Year Treasury Bond ETF': 'U.S. LT Treas',
                    'Cash': 'Cash',
                    'SP500': 'U.S. Stocks (S&P 500)',
                    'US_REITs': 'U.S. Real Estate (REITs)', 
                    'US_LT_Treas': 'U.S. LT Treas',
                    'Intl_Dev_Stocks': 'International Developed Stocks',
                    'EM_Stocks': 'Emerging Market Stocks',
                    'Commodities': 'Commodities',
                    'Gold': 'Gold'
                }
                
                latest_allocation['display_name'] = latest_allocation['asset_name'].map(asset_name_map).fillna(latest_allocation['asset_name'])
                
                # Combine allocations that map to the same display name (e.g., Cash + US_LT_Treas both become "Cash")
                latest_allocation = latest_allocation.groupby('display_name')['allocation_percentage'].sum().reset_index()
                
                display_allocation = latest_allocation[['display_name', 'allocation_percentage']].copy()
                display_allocation = display_allocation.sort_values('allocation_percentage', ascending=False)
                display_allocation.columns = ['Asset Class', 'Allocation (%)']
                display_allocation['Allocation (%)'] = (display_allocation['Allocation (%)'] * 100).round(2)  # Convert to percentage
                
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
        
        returns_data = data['monthly_returns']
        returns_col = data.get('returns_column', 'ga_returns_net')
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Monthly Return", f"{returns_data[returns_col].mean()*100:.2f}%")
        with col2:
            st.metric("Best Month", f"{returns_data[returns_col].max()*100:.2f}%")
        with col3:
            st.metric("Worst Month", f"{returns_data[returns_col].min()*100:.2f}%")
        with col4:
            st.metric("Win Rate", f"{(returns_data[returns_col] > 0).mean()*100:.1f}%")
    
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
        st.plotly_chart(create_benchmark_performance_chart(data, strategy), use_container_width=True)
        
        
    
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
                display_name = asset_name_map.get(asset, asset)
                st.metric(display_name, f"{contrib:.2f}%")
        
        with col2:
            st.markdown("### Detractors")
            detractors = attribution_summary.tail(4)
            for asset, contrib in detractors.items():
                display_name = asset_name_map.get(asset, asset)
                st.metric(display_name, f"{contrib:.2f}%")
    
if __name__ == "__main__":
    main()
