# RQA Portfolio Dashboard

A professional portfolio analytics dashboard built with Streamlit for the RQA Global Adaptive Strategy. This application provides comprehensive performance analysis, benchmark comparisons, and interactive visualizations for portfolio management.

## Features

### 📊 Strategy Fact Sheet
- Professional strategy overview with key performance metrics
- Growth of $100,000 visualization
- Current asset allocation pie chart
- Benchmark performance comparison table
- Risk metrics and portfolio statistics

### 📈 Interactive Analytics Dashboard
- **Current Allocation**: Real-time portfolio allocation visualization
- **Trailing 12M Allocations**: Historical allocation stacked bar charts
- **Monthly Returns**: Monthly performance with rolling volatility analysis
- **Benchmark Performance**: Multi-metric comparison across portfolios (2x2 chart layout)
- **Attribution Analysis**: Asset contribution analysis over trailing 12 months

## Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **Data Visualization**: Plotly for interactive charts
- **Database**: SQLite for data storage
- **Data Processing**: Pandas and NumPy
- **Styling**: Custom blue theme with professional styling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rqa-portfolio-dashboard.git
cd rqa-portfolio-dashboard
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Setting Up the Database

1. Place your `GAM.xlsx` file in the project directory
2. Run the database creation script:
```bash
python excel_to_database.py
```

This will create `portfolio_data.db` with all necessary tables:
- `monthly_returns`: Historical performance data
- `gam_allocations`: Asset allocation history
- `gam_attribution`: Performance attribution data
- `benchmark_performance`: Calculated performance metrics

### Running the Dashboard

```bash
streamlit run portfolio_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Database Export (Optional)

Export database tables for analysis:
```bash
python export_database.py
```

## Project Structure

```
rqa-portfolio-dashboard/
├── portfolio_dashboard.py      # Main Streamlit dashboard
├── excel_to_database.py       # Database creation from Excel
├── export_database.py         # Database export utilities
├── requirements.txt           # Python dependencies
├── portfolio_data.db          # SQLite database (generated)
├── GAM.xlsx                   # Source data file
└── README.md                  # This file
```

## Dashboard Sections

### Strategy Fact Sheet
- Comprehensive overview of the RQA Global Adaptive Strategy
- Key performance metrics with professional styling
- Growth visualization and current allocation
- Risk-adjusted performance metrics

### Interactive Analytics
- **Current Allocation**: Live portfolio composition
- **12M Allocations**: Historical allocation trends
- **Monthly Returns**: Performance analysis with volatility
- **Benchmark Performance**: 4-panel comparison (Returns, Risk, Sharpe, Beta)
- **Attribution**: Asset-level contribution analysis

## Data Schema

### Monthly Returns Table
- Date, GAM returns (gross/net), benchmark returns
- Portfolio variations (50/50, 60/40, 70/30)

### Allocations Table
- Historical asset allocation percentages
- Asset mapping for consistent naming

### Attribution Table
- Monthly asset-level performance attribution
- Contribution analysis for portfolio returns

### Benchmark Performance Table
- Calculated metrics: YTD, 1Y, 5Y, Since Inception
- Risk metrics: Volatility, Sharpe Ratio, Beta

## Styling Features

- **RQA Blue Theme**: Professional color scheme
- **Pill-style Tabs**: Modern navigation interface
- **Responsive Design**: Works on desktop and mobile
- **Interactive Charts**: Hover details and zoom functionality
- **Professional Layout**: Clean, institutional-grade presentation

## Performance Calculations

- **Returns**: Compound annual growth rates
- **Volatility**: Annualized standard deviation
- **Sharpe Ratio**: Risk-adjusted returns (RF=0)
- **Beta**: Correlation to S&P 500 benchmark
- **Attribution**: Asset-level contribution to returns

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact the RQA team.

---

**Disclaimer**: This dashboard is for informational purposes only. Past performance does not guarantee future results. All investments involve risk of loss.