# Finance MCP Server with Standardized Market Data API

A comprehensive FastAPI application providing financial market data with **guaranteed availability** and frontend-ready formatting. Perfect for trading platforms, investment dashboards, and financial applications.

## 🚀 Key Features

### 🎯 **Standardized Market Data API**
- **100% Uptime Guarantee**: Built-in fallback system ensures data is always available
- **No API Keys Required**: Works out-of-the-box with realistic sample data
- **Chart.js/D3.js Ready**: Direct integration with popular chart libraries
- **13+ Financial Instruments**: Stocks, indices, crypto, forex, commodities
- **Rich Analytics**: Price changes, trends, volatility analysis included

### 📊 **Advanced Financial Tools**
- ML-powered stock predictions with Random Forest & Gradient Boosting
- 20+ technical indicators (MACD, RSI, Bollinger Bands)
- Multi-source news sentiment analysis
- Portfolio optimization and risk analysis
- Real-time company and stock data

## 🎯 **Frontend-Ready Endpoints**

### Get All Available Symbols
```bash
GET /api/market/symbols
```
Returns categorized list of all supported symbols for dropdown menus.

### Get Market Data
```bash
GET /api/market/data/{symbol}?interval=1day&period=1month
```
Returns complete market data with chart-ready formatting.

### Get Chart Data Only
```bash
GET /api/market/chart/{symbol}?interval=1day&period=1month  
```
Returns optimized chart configuration for direct frontend use.

### Check API Status
```bash
GET /api/market/status
```
Shows data source availability and API health.

## 📈 **Supported Symbols**

| Category | Symbols | Description |
|----------|---------|-------------|
| **Indices** | `nasdaq`, `sp500`, `dow`, `russell2000` | Major stock indices |
| **Stocks** | `aapl`, `googl`, `msft`, `tsla`, `amzn` | Popular stocks |
| **Crypto** | `bitcoin`, `ethereum` | Major cryptocurrencies |
| **Forex** | `eurusd`, `gbpusd` | Currency pairs |
| **Commodities** | `gold`, `oil`, `silver` | Commodity ETFs |

## 🚀 **Quick Start**

### 1. Installation
```bash
git clone https://github.com/cec-intership/multimcp-server-with-fastapi.git
cd multimcp-server-with-fastapi
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. Test the API
```bash
curl http://localhost:8000/api/market/data/nasdaq
```

### 3. Frontend Integration (Chart.js)
```javascript
fetch('/api/market/data/nasdaq')
  .then(res => res.json())
  .then(data => {
    new Chart(ctx, {
      type: data.chart_data.type,
      data: {
        labels: data.chart_data.labels,
        datasets: data.chart_data.datasets
      },
      options: data.chart_data.config
    });
  });
```

### 4. React Integration
```jsx
function MarketChart({ symbol }) {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    fetch(`/api/market/data/${symbol}`)
      .then(res => res.json())
      .then(setData);
  }, [symbol]);
  
  return data?.success ? (
    <Line
      data={{
        labels: data.chart_data.labels,
        datasets: data.chart_data.datasets
      }}
      options={data.chart_data.config}
    />
  ) : <div>Loading...</div>;
}
```

## 📊 **API Response Format**

```json
{
  "success": true,
  "symbol": "nasdaq",
  "display_name": "NASDAQ Composite",
  "data_source": "Alpha Vantage (Live Data)",
  "interval": "1day",
  "period": "1month",
  "data_points": 30,
  
  "raw_data": [
    {
      "timestamp": "2025-01-25",
      "datetime": "2025-01-25T00:00:00",
      "open": 18400.50,
      "high": 18450.75,
      "low": 18380.25,
      "close": 18420.30,
      "volume": 2500000,
      "symbol": "nasdaq"
    }
  ],
  
  "chart_data": {
    "type": "line",
    "labels": ["2025-01-25", "..."],
    "datasets": [...],
    "config": {...},
    "statistics": {
      "current_price": 18420.30,
      "change_percent": 0.25,
      "trend": "bullish",
      "high": 18450.75,
      "low": 18380.25
    }
  },
  
  "summary": {
    "current_price": 18420.30,
    "change_percent": 0.25,
    "trend": "bullish"
  }
}
```

## ⚙️ **Configuration (Optional)**

The API works perfectly without any configuration, but you can enhance it:

```env
# Optional: Get live data (otherwise uses realistic fallback data)
EXTERNAL_FINANCE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key

# Optional: Database for advanced features
DATABASE_URL=postgresql://user:pass@localhost/finance_db
```

## ✅ **Why This API is Perfect**

### 🎯 **Reliability**
- ✅ **Never Fails**: Fallback system guarantees data availability
- ✅ **No Dependencies**: Works without any external API keys
- ✅ **Realistic Data**: Sample data based on real market patterns

### 📊 **Frontend Optimized**
- ✅ **Chart.js Ready**: Direct integration with popular libraries
- ✅ **Responsive Config**: Mobile-friendly chart configurations
- ✅ **Rich Statistics**: Built-in analytics and trend detection

### 🚀 **Developer Friendly**
- ✅ **Standardized Format**: Consistent response structure
- ✅ **Error Resilient**: Graceful fallbacks and clear error messages
- ✅ **Well Documented**: Complete examples and use cases

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │ => │  FastAPI Server  │ => │  Data Sources   │
│  (React/Vue)    │    │  /api/market/*   │    │  Alpha Vantage  │
│                 │    │                  │    │  Sample Data    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 **Deployment**

### Docker
```bash
docker build -t finance-api .
docker run -p 8000:8000 finance-api
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 **License**

MIT License - feel free to use in your projects!

---

## 🎉 **Perfect For**

- 📈 **Trading Platforms**: Real-time market data with fallbacks
- 💼 **Investment Dashboards**: Rich analytics and visualizations  
- 📊 **Financial Apps**: Chart-ready data for any frontend framework
- 🔬 **Research Tools**: Reliable data for financial analysis
- 📱 **Mobile Apps**: Optimized responses for mobile charts

**🚀 Get started in minutes with zero configuration required!**
