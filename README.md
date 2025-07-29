# Finance MCP Server with FastAPI 📈

A comprehensive financial analysis server with **advanced ML predictions**, **technical analysis**, **sentiment analysis**, and **real-time market data** through optimized database caching.

## 🎯 **Key Features**

### **Advanced Financial Analysis**
- **🤖 ML Predictions**: Multiple ML models (Random Forest, Gradient Boosting, Linear Regression)
- **📊 Technical Analysis**: 15+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **💭 Sentiment Analysis**: News-based market sentiment with real-time insights
- **📈 Portfolio Management**: Complete portfolio tracking and analysis tools
- **🎯 Comprehensive Testing**: Full test coverage for all prediction and analysis tools

### **Smart Data Management**
- **Local Database Integration**: PostgreSQL with 2+ years of historical data
- **Comprehensive Data Import**: 60+ symbols across multiple categories
- **Multi-Source Data**: yfinance, Alpha Vantage, and market APIs
- **Background Data Refresh**: Automated data updates and validation
- **Vietnamese & International Markets**: Complete coverage of major markets

### **Production-Ready Architecture**
- **FastAPI Server**: High-performance async API with <100ms response times
- **MCP Server Integration**: 30+ financial analysis tools
- **Comprehensive Testing**: Automated test suite for all components
- **Error Handling**: Robust error handling and logging

## 🚀 **Quick Start**

### **1. Complete Setup (Recommended)**
```bash
# Run the complete setup process
python master_runner.py

# This will:
# - Install all dependencies
# - Import comprehensive financial data (60+ symbols)
# - Run full test suite (17 test categories)
# - Generate detailed reports
```

### **2. Manual Setup**
```bash
# Install dependencies
pip install fastapi uvicorn psycopg2-binary python-dotenv pandas yfinance scikit-learn matplotlib

# Set up environment variables
cp .env.example .env
# Edit .env with your database URI

# Import data only
python run_data_import.py

# Run quick tests
python quick_test.py

# Run comprehensive tests
python comprehensive_test_suite.py
```

### **3. Start the Server**
```bash
# Start the FastAPI server
python main.py

# Or use VS Code task
# Ctrl+Shift+P -> "Tasks: Run Task" -> "Start MCP FastAPI Server"
```

### **4. Verify Setup**
```bash
# Check API health
curl http://127.0.0.1:8000/api/health

# Test prediction endpoints
curl http://127.0.0.1:8000/api/stock/AAPL
curl http://127.0.0.1:8000/api/companies

# Check server status
curl http://127.0.0.1:8000/api/admin/data-status
```

## 🧪 **Testing & Validation**

### **Comprehensive Test Suite**
The system includes extensive testing for all financial analysis tools:

```bash
# Run all tests (17 categories)
python comprehensive_test_suite.py

# Categories tested:
# - ML Prediction Models (Random Forest, Gradient Boosting, Linear)
# - Technical Analysis (RSI, MACD, Bollinger Bands, volatility)
# - Sentiment Analysis (news sentiment, market alerts)
# - Financial Calculations (returns, volume analysis, comparisons)
# - Market Data (real-time data, chart data, market overview)
# - Visualization (price charts, volume charts)
# - Portfolio Management (add/remove holdings, tracking)
```

### **Test Symbols**
The system tests across multiple categories:
- **High Volume**: AAPL, GOOGL, MSFT, TSLA, NVDA
- **Market Indices**: SPY, QQQ, DIA
- **Commodities**: GLD, SLV, USO
- **Crypto ETFs**: BITO, COIN
- **Volatile Stocks**: GME, AMC, PLTR

## 📊 **API Usage Examples**

### **Core Market Data**
```javascript
// Vietnamese market data
GET /api/market-data/VN-INDEX

// Individual stock data
GET /api/stock/VCB

// Watchlist stocks (6 Vietnamese stocks)
GET /api/watchlist-stocks

// Historical chart data
GET /api/chart-data/VCB?period=3months
```

### **Advanced Analysis**
```javascript
// Technical analysis with RSI, MACD, Bollinger Bands
GET /api/technical-analysis/VCB

// Compare two stocks
GET /api/compare-stocks/VCB/VIC

// Volatility and risk analysis
GET /api/volatility/VCB

// Company information
GET /api/company-info/VCB

// AI price predictions
GET /api/predictions/VCB

// Portfolio performance
GET /api/portfolio
```

### **Admin Functions**
```javascript
// Trigger data import for all stocks
POST /api/admin/import-data

// Import specific stock
POST /api/admin/import-stock/VCB

// Check data freshness
GET /api/admin/data-status
```

## � **Database Integration**

The system uses your existing PostgreSQL schema:

### **Required Tables**
- `COMPANY`: Company information (symbol, name, sector, industry, etc.)
- `STOCK_PRICE`: Historical OHLC data with volume
- `data_import_log`: Import tracking (created automatically)

### **Performance Benefits**
- **Response Time**: <100ms (vs 2-5 seconds with external APIs)
- **API Calls**: 99% reduction in external API usage
- **Reliability**: No rate limiting or external API downtime
- **Cost**: Minimal API quota consumption

## � **Environment Variables**

```env
# Database (Required)
FINANCE_DB_URI=postgresql://username:password@localhost:5432/finance_db

# Optional: External API keys for data sources
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key

# Server Configuration
HOST=127.0.0.1
PORT=8000
```

## 🎯 **Data Management**

### **Manual Data Import**
```bash
# Import all Vietnamese stocks
curl -X POST http://127.0.0.1:8000/api/admin/import-data

# Import specific stock
curl -X POST http://127.0.0.1:8000/api/admin/import-stock/VCB

# Check which stocks need updates
curl http://127.0.0.1:8000/api/admin/data-status
```

### **Automatic Background Updates**
The system automatically refreshes stale data when:
- Data is older than 24 hours
- Frontend requests data for a symbol
- Manual refresh is triggered via admin endpoints

## 🏗️ **Architecture**

```
├── main.py                 # FastAPI application entry point
├── api/
│   └── direct_finance_api.py   # Direct API endpoints with database integration
├── mcp_servers/            # MCP server modules for advanced analysis
│   ├── finance_market_data.py
│   ├── finance_analysis_and_predictions.py
│   ├── finance_calculations.py
│   └── ...
├── utils/
│   └── data_import.py      # Database integration and data import
└── README.md              # This file
```

## 📈 **Supported Stocks**

### **Vietnamese Market**
VCB, VIC, VHM, HPG, TCB, MSN, FPT, GAS, CTG, MWG, BID, ACB, VPB, POW, VRE, PLX, SAB, MBB

### **International Market**
AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, NFLX, BABA, TSM

## 🔍 **Monitoring & Troubleshooting**

### **Check System Status**
```bash
# API health check
curl http://127.0.0.1:8000/api/health

# Database status
curl http://127.0.0.1:8000/api/admin/data-status
```

### **Common Issues**

**1. No data for a symbol**
```bash
# Import data for specific symbol
curl -X POST http://127.0.0.1:8000/api/admin/import-stock/SYMBOL
```

**2. Database connection issues**
- Check `FINANCE_DB_URI` in `.env`
- Ensure PostgreSQL is running
- Verify database contains COMPANY and STOCK_PRICE tables

**3. Slow responses**
- Check if data is fresh with `/api/admin/data-status`
- Import fresh data with `/api/admin/import-data`

## 🚀 **Production Deployment**

### **Scheduled Data Updates**
Set up a cron job for daily data updates:
```bash
# Add to crontab (daily at 6 AM)
0 6 * * * curl -X POST http://localhost:8000/api/admin/import-data
```

### **Performance Optimization**
1. **Database Indexing**: Ensure indexes on `symbol` and `date` columns
2. **Connection Pooling**: Use connection pooling for high-traffic scenarios
3. **Load Balancing**: Use multiple server instances with shared database

---

## 📝 **API Endpoints Summary**

| Endpoint | Method | Description | Response Time |
|----------|--------|-------------|---------------|
| `/api/health` | GET | System health check | <10ms |
| `/api/market-data/{symbol}` | GET | Market data | <100ms |
| `/api/watchlist-stocks` | GET | Vietnamese watchlist | <200ms |
| `/api/stock/{symbol}` | GET | Individual stock data | <50ms |
| `/api/technical-analysis/{symbol}` | GET | Technical indicators | <500ms |
| `/api/company-info/{symbol}` | GET | Company information | <100ms |
| `/api/admin/import-data` | POST | Import all stock data | 30-60s |
| `/api/admin/data-status` | GET | Data freshness status | <500ms |

**🎉 Your finance API is now optimized for high-performance production use!**

**Replace mockMarketData:**
```javascript
// Instead of: import { mockMarketData } from './mockData'
const marketData = await fetch('/api/market-data/VN-INDEX').then(r => r.json());
```

**Replace mockWatchlistStocks:**
```javascript
// Instead of: import { mockWatchlistStocks } from './mockData'
const watchlistStocks = await fetch('/api/watchlist-stocks').then(r => r.json());
```

**Replace mockNewsItems:**
```javascript
// Instead of: import { mockNewsItems } from './mockData'
const newsItems = await fetch('/api/news?limit=10').then(r => r.json());
```

### MCP Services

| Service | Endpoint | Purpose |
|---------|----------|---------|
| **Market Data** | `/finance_market_data/` | Real-time market data with auto-discovery |
| **Predictions** | `/finance_analysis_and_predictions/` | ML-based analysis and forecasting |
| **Portfolio** | `/finance_portfolio/` | Portfolio management and optimization |
| **Calculations** | `/finance_calculations/` | Advanced financial calculations |
| **News & Insights** | `/finance_news_and_insights/` | News aggregation and sentiment |
| **Visualization** | `/finance_plotting/` | Chart generation and plotting |
| **Data Ingestion** | `/finance_data_ingestion/` | External data import tools |
| **Company DB** | `/finance_db_company/` | Company information database |
| **Stock Price DB** | `/finance_db_stock_price/` | Historical price data storage |
| **Utilities** | `/echo/`, `/math/`, `/user_db/` | Testing and utility functions |

## 💡 Usage Examples

### 1. Get Real-time Market Data (Direct API)

```javascript
// Frontend JavaScript - replace mockMarketData
const getMarketData = async (symbol = 'VN-INDEX') => {
  const response = await fetch(`/api/market-data/${symbol}`);
  const marketData = await response.json();
  
  console.log(`${marketData.company}: $${marketData.currentPrice}`);
  console.log(`Change: $${marketData.close - marketData.prevClose}`);
  
  return marketData; // Compatible with MarketData interface
};
```

### 2. Get Watchlist Stocks (Replace Mock Data)

```javascript
// Frontend JavaScript - replace mockWatchlistStocks
const getWatchlistStocks = async () => {
  const response = await fetch('/api/watchlist-stocks');
  const stocks = await response.json();
  
  stocks.forEach(stock => {
    console.log(`${stock.symbol}: $${stock.price} (${stock.changePercent:+.2f}%)`);
  });
  
  return stocks; // Compatible with StockData[] interface
};
```

### 3. Get Financial News (Replace Mock Data)

```javascript
// Frontend JavaScript - replace mockNewsItems
const getFinancialNews = async (limit = 10) => {
  const response = await fetch(`/api/news?limit=${limit}`);
  const newsItems = await response.json();
  
  newsItems.forEach(news => {
    console.log(`${news.title} - ${news.source}`);
  });
  
  return newsItems; // Compatible with NewsItem[] interface
};
```

### 4. Get Chart Data for Visualization

```javascript
// Frontend JavaScript - get chart data for any symbol
const getChartData = async (symbol, period = '1month') => {
  const response = await fetch(`/api/chart-data/${symbol}?period=${period}`);
  const chartData = await response.json();
  
  // Direct Chart.js integration
  new Chart(document.getElementById('stockChart'), {
    type: 'line',
    data: {
      labels: chartData.map(point => point.time),
      datasets: [{
        label: symbol,
        data: chartData.map(point => point.price),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }]
    },
    options: { responsive: true }
  });
  
  return chartData;
};
```

### 5. Original MCP Server Usage (Advanced)

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/finance_market_data/",
    json={
        "method": "get_market_data",
        "params": {
            "symbol": "AAPL",
            "interval": "1day",
            "period": "1month",
            "format_for_chart": True
        }
    }
)

data = response.json()
print(f"Apple current price: ${data['chart_data']['statistics']['current_price']}")
```

### 2. Get AI Price Prediction

```python
response = requests.post(
    "http://127.0.0.1:8000/finance_analysis_and_predictions/",
    json={
        "method": "predict_stock_price",
        "params": {
            "symbol": "TSLA",
            "days_ahead": 30,
            "model_type": "random_forest"
        }
    }
)

prediction = response.json()
print(f"Tesla 30-day prediction: ${prediction['prediction']['predicted_price']}")
```

### 3. Create and Analyze Portfolio

```python
# Create portfolio
response = requests.post(
    "http://127.0.0.1:8000/finance_portfolio/",
    json={
        "method": "create_portfolio",
        "params": {
            "name": "Tech Portfolio",
            "positions": [
                {"symbol": "AAPL", "shares": 100, "purchase_price": 150.00},
                {"symbol": "GOOGL", "shares": 50, "purchase_price": 2800.00}
            ]
        }
    }
)

portfolio = response.json()
print(f"Portfolio value: ${portfolio['total_value']}")
print(f"Total return: {portfolio['total_return']:.2%}")
```

### 4. Frontend Integration (JavaScript)

```javascript
// Fetch market data for charts
async function loadStockChart(symbol) {
    const response = await fetch('/finance_market_data/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            method: 'get_market_data',
            params: {
                symbol: symbol,
                interval: '1day',
                period: '3months',
                format_for_chart: true
            }
        })
    });
    
    const data = await response.json();
    
    // Direct Chart.js integration
    new Chart(document.getElementById('stockChart'), {
        type: 'line',
        data: data.chart_data,
        options: { responsive: true }
    });
}

loadStockChart('AAPL');
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│                     │    │                      │    │                     │
│   FastAPI Server    │────│   MCP Integration    │────│   Finance Modules   │
│                     │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
           │                           │                           │
           │                           │                           │
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│                     │    │                      │    │                     │
│   Static Dashboard  │    │   SQLite Database    │    │   External APIs     │
│                     │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Data Flow

1. **Request Processing**: FastAPI receives HTTP requests
2. **MCP Routing**: Requests routed to appropriate MCP server
3. **Data Sources**: Multiple APIs queried with priority fallbacks
4. **Database Layer**: Results cached in SQLite for performance
5. **Response Formatting**: Data formatted for charts/frontend consumption
6. **Client Response**: JSON response with complete data and metadata

### Key Technologies

- **FastAPI**: High-performance Python web framework
- **MCP (Model Context Protocol)**: AI assistant integration
- **SQLite**: Lightweight database for caching and storage
- **scikit-learn**: Machine learning for predictions
- **pandas/numpy**: Data processing and analysis
- **yfinance**: Yahoo Finance API integration
- **Chart.js**: Frontend charting (compatible output)

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys (get from respective providers)
EXTERNAL_FINANCE_API_KEY=your_alpha_vantage_key    # Alpha Vantage (primary)
FINNHUB_API_KEY=your_finnhub_key                   # Finnhub (secondary) 
NEWS_API_KEY=your_news_api_key                     # News API (optional)

# Server Configuration
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=info

# Database
DB_PATH=./data/finance.db

# Cache Settings
CACHE_TTL_MARKET_DATA=3600    # 1 hour
CACHE_TTL_NEWS=900            # 15 minutes
CACHE_TTL_COMPANY=86400       # 24 hours
```

### Getting API Keys

1. **Alpha Vantage** (Required for live data)
   - Visit: https://www.alphavantage.co/support/#api-key
   - Free tier: 500 requests/day
   - Premium: Up to 1200 requests/minute

2. **Finnhub** (Optional, additional coverage)
   - Visit: https://finnhub.io/register
   - Free tier: 60 requests/minute
   - Premium: Up to 600 requests/minute

3. **News API** (Optional, for news features)
   - Visit: https://newsapi.org/register
   - Free tier: 1000 requests/day
   - Premium: Up to 100,000 requests/day

## 📈 Supported Assets

### Stocks & Indices
- **Major Stocks**: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, etc.
- **Indices**: NASDAQ, S&P 500, Dow Jones, Russell 2000
- **International**: Any ticker symbol via auto-discovery

### Cryptocurrencies
- Bitcoin (BTC-USD)
- Ethereum (ETH-USD)
- Major altcoins via symbol lookup

### Forex Pairs
- EUR/USD, GBP/USD, USD/JPY
- Major and minor currency pairs

### Commodities
- Gold (GLD ETF)
- Oil (USO ETF) 
- Silver (SLV ETF)
- Other commodity ETFs

### Auto-Discovery
The system can automatically discover and add support for any publicly traded symbol by:
1. Querying multiple data sources
2. Extracting company information
3. Storing in local database
4. Providing cached access for future requests

## 🧪 Testing

### Run Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Test Direct Finance API
```bash
# Test market data
curl http://127.0.0.1:8000/api/market-data/VN-INDEX

# Test watchlist stocks
curl http://127.0.0.1:8000/api/watchlist-stocks

# Test individual stock
curl http://127.0.0.1:8000/api/stock/AAPL

# Test financial news
curl http://127.0.0.1:8000/api/news?limit=5

# Test chart data
curl http://127.0.0.1:8000/api/chart-data/AAPL
```

### Run Automated Tests
```bash
python test_foxstocks_api.py
```

### Test MCP Endpoints (Advanced)
```bash
# Original MCP server endpoints are still available
curl -X POST http://127.0.0.1:8000/finance_market_data/ \
  -H "Content-Type: application/json" \
  -d '{
    "method": "get_market_data",
    "params": {
      "symbol": "AAPL",
      "interval": "1day",
      "period": "1week"
    }
  }'
```

## 🚀 Deployment

### Production Setup

1. **Use production WSGI server**
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **Environment variables**
   ```bash
   export ENVIRONMENT=production
   export LOG_LEVEL=warning
   export HOST=0.0.0.0
   export PORT=8000
   ```

3. **Reverse proxy (nginx)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

```bash
docker build -t finance-mcp-server .
docker run -p 8000:8000 --env-file .env finance-mcp-server
```

## 📚 Documentation

### API Documentation
- **Interactive Docs**: `/docs` - Swagger UI with live testing
- **Alternative Docs**: `/redoc` - ReDoc documentation
- **API Manual**: `API_MANUAL.md` - Comprehensive usage guide

### Architecture Documentation
- **MCP Integration**: How Model Context Protocol is implemented
- **Data Sources**: External API integration and fallback strategies  
- **Database Schema**: SQLite table structures and relationships
- **Caching Strategy**: Performance optimization through intelligent caching

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Setup

```bash
# Clone repository
git clone https://github.com/cec-intership/multimcp-server-with-fastapi.git
cd multimcp-server-with-fastapi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Start development server
python main.py
```

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all public methods
- Write tests for new features
- Update documentation for API changes

## 🐛 Troubleshooting

### Common Issues

**1. No market data returned**
- Check API key configuration in `.env`
- Verify internet connection
- Check `/status` endpoint for API key status

**2. Slow response times**
- Check external API rate limits
- Review caching configuration
- Monitor database performance

**3. Import errors**
- Ensure all dependencies are installed
- Check Python version compatibility
- Verify virtual environment activation

**4. Database errors**
- Check database file permissions
- Verify SQLite installation
- Check disk space availability

### Getting Help

1. **Check server status**: `GET /health`
2. **Review logs**: Server console output
3. **Test connectivity**: `GET /echo/`
4. **Verify configuration**: `GET /status`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Alpha Vantage** for comprehensive financial data API
- **Finnhub** for additional market data coverage
- **Yahoo Finance** for supplementary data sources
- **FastAPI** for the excellent web framework
- **Model Context Protocol** for AI assistant integration
- **Chart.js** for frontend charting inspiration

## 📞 Support

### Community
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides at `/docs`
- **Examples**: Sample code in `examples/` directory

### Commercial Support
For enterprise deployments and custom integrations, contact the development team.

---

**🚀 Start building powerful financial applications with the Finance MCP Server today!**

*Built with ❤️ by the CEC Internship Team*
