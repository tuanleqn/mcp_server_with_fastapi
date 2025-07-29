# 🏦 Finance MCP Server

> **A comprehensive financial analysis platform with Model Context Protocol (MCP) integration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP-1.0+-purple.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

The Finance MCP Server is a powerful, production-ready financial analysis platform that combines real-time market data, machine learning predictions, portfolio management, and comprehensive financial tools. Built with FastAPI and Model Context Protocol (MCP), it provides both REST API endpoints and MCP-compatible interfaces for seamless integration with AI assistants and financial applications.

### ✨ Key Features

- 📈 **Real-time Market Data** - Stocks, indices, crypto, forex, and commodities
- 🤖 **AI-Powered Predictions** - ML-based price forecasting and trend analysis
- 💼 **Portfolio Management** - Advanced portfolio tracking and optimization
- 📊 **Technical Analysis** - 20+ technical indicators and chart patterns
- 📰 **News & Sentiment** - Financial news aggregation with sentiment analysis
- 📉 **Interactive Charts** - Chart.js/D3.js compatible visualizations
- 💾 **Database Integration** - SQLite with auto-discovery and caching
- 🔄 **Multi-Source Data** - Alpha Vantage, Finnhub, Yahoo Finance integration
- 🚀 **100% Uptime** - Smart fallback system ensures data availability
- 🔧 **Developer Friendly** - Comprehensive API documentation and examples

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or uv package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/cec-intership/multimcp-server-with-fastapi.git
   cd multimcp-server-with-fastapi
   ```

2. **Install dependencies**
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # Or using uv (recommended)
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your API keys:
   ```env
   # Required for live market data
   EXTERNAL_FINANCE_API_KEY=your_alpha_vantage_key
   
   # Optional for additional data sources
   FINNHUB_API_KEY=your_finnhub_key
   NEWS_API_KEY=your_news_api_key
   ```

4. **Start the server**
   ```bash
   python main.py
   ```

5. **Access the application**
   - **Web Dashboard:** http://127.0.0.1:8000
   - **API Documentation:** http://127.0.0.1:8000/docs
   - **Health Check:** http://127.0.0.1:8000/health

## 📊 API Overview

### Core Endpoints

| Endpoint | Description | Example |
|----------|-------------|---------|
| `GET /` | Dashboard or API info | `curl http://127.0.0.1:8000/` |
| `GET /docs` | Interactive API documentation | - |
| `GET /health` | Health check and status | `curl http://127.0.0.1:8000/health` |
| `GET /status` | Detailed server configuration | `curl http://127.0.0.1:8000/status` |
| `GET /tools` | Available MCP tools list | `curl http://127.0.0.1:8000/tools` |

### Direct Finance API (No Authentication Required)

| Endpoint | Method | Description | Response Format |
|----------|--------|-------------|-----------------|
| `/api/market-data/{symbol}` | GET | Get market data (OHLC) | `MarketData` interface |
| `/api/watchlist-stocks` | GET | Get watchlist stocks | `StockData[]` interface |
| `/api/stock/{symbol}` | GET | Get individual stock data | `StockData` interface |
| `/api/chart-data/{symbol}` | GET | Get chart time series data | `ChartDataPoint[]` |
| `/api/news` | GET | Get financial news | `NewsItem[]` interface |
| `/api/predictions/{symbol}` | GET | Get AI price predictions | Prediction object |
| `/api/portfolio` | GET | Get portfolio performance | Portfolio object |

**Frontend Integration Examples:**

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
