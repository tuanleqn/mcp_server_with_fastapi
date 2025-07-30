# Finance API Documentation

## 🚀 Overview

The Finance API provides comprehensive financial data and analysis tools for frontend applications. The API is built with a **modular architecture** consisting of 3 main components:

- **Core Endpoints**: Basic market data, stocks, charts, news
- **Advanced Endpoints**: Portfolio management, technical analysis, predictions
- **Finance Models**: Pydantic data models and type definitions

## 📊 Architecture

### Modular Structure
```
api/
├── direct_finance_api.py      # Main router combining all modules
├── core_endpoints.py          # Basic market data endpoints
├── advanced_endpoints.py      # Advanced analytics endpoints
└── finance_models.py          # Pydantic models and schemas
```

### Data Flow Strategy
1. **Local Database First** - Fastest access to cached data
2. **MCP Database Servers** - Direct database access via MCP
3. **MCP External APIs** - Real-time data when needed  
4. **Fallback Data** - Ensures API never fails

## 🏗️ API Information

- **Version**: 4.0.0
- **Architecture**: Modular
- **Total Endpoints**: 37+
- **MCP Servers Integrated**: 13
- **Asset Classes Supported**: 4
- **Symbols Tracked**: 21 reliable symbols

## 📈 Asset Classes

### Big Tech Stocks (10)
- AAPL - Apple Inc.
- GOOGL - Alphabet Inc. (Google)
- MSFT - Microsoft Corporation
- AMZN - Amazon.com Inc.
- META - Meta Platforms Inc.
- TSLA - Tesla Inc.
- NVDA - NVIDIA Corporation
- NFLX - Netflix Inc.
- CRM - Salesforce Inc.
- ORCL - Oracle Corporation

### Market Indices (5)
- SPY - S&P 500 ETF
- QQQ - Nasdaq 100 ETF
- DIA - Dow Jones Industrial Average ETF
- VTI - Total Stock Market ETF
- IWM - Russell 2000 ETF

### Commodities (4)
- GLD - Gold ETF
- SLV - Silver ETF
- USO - Oil ETF
- UNG - Natural Gas ETF

### Crypto ETFs (2)
- BITO - Bitcoin Strategy ETF
- ETHE - Ethereum Trust

## 🔌 Core Endpoints

### Market Data
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/market-overview` | GET | Comprehensive market dashboard |
| `/api/market-data/{symbol}` | GET | Market data for specific symbol |
| `/api/market/status` | GET | Current market status and trading hours |

### Stock Data
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/watchlist-stocks` | GET | Top 8 mixed assets for watchlist |
| `/api/stock/{symbol}` | GET | Individual stock data |
| `/api/chart-data/{symbol}` | GET | Historical price charts |
| `/api/company-info/{symbol}` | GET | Detailed company information |

### News & Sentiment
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/news` | GET | Financial news feed |
| `/api/market-sentiment` | GET | Overall market sentiment analysis |

### Asset Categories
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/crypto-data` | GET | Cryptocurrency ETF data |
| `/api/commodities-data` | GET | Commodities (Gold, Silver, Oil, Gas) |
| `/api/indices-data` | GET | Major market indices |

## 🧠 Advanced Endpoints

### Technical Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/technical-analysis/{symbol}` | GET | Technical indicators and analysis |
| `/api/predictions/{symbol}` | GET | AI price predictions |
| `/api/risk-analysis/{symbol}` | GET | Comprehensive risk assessment |
| `/api/sector-analysis` | GET | Sector performance analysis |

### Portfolio Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/portfolio` | GET | Portfolio performance data |
| `/api/portfolio/analysis` | GET | Detailed portfolio analysis |
| `/api/portfolio/optimization` | GET | Portfolio optimization suggestions |
| `/api/portfolio/risk-return` | GET | Risk-return analysis |

### Financial Calculations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stock-return/{symbol}` | GET | Calculate stock returns between dates |
| `/api/trading-volume/{symbol}` | GET | Trading volume analysis |
| `/api/calculations/compound-return` | GET | Compound return calculations |

### Market Intelligence
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/market/correlation-matrix` | GET | Multi-symbol correlation analysis |
| `/api/search/symbols` | GET | Advanced symbol search |

### User & Database Access
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/user/{user_name}` | GET | User data by name |
| `/api/user-by-email/{user_email}` | GET | User data by email |
| `/api/database/stock-price/{symbol}` | GET | Direct database price access |
| `/api/database/company/{symbol}` | GET | Direct database company info |

### Mathematical Operations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/math/add` | GET | Mathematical addition |

### System Health & Admin
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | API health check |
| `/api/health/modular` | GET | Modular architecture health |
| `/api/data-ingestion/status` | GET | Data ingestion status |
| `/api/data-ingestion/health` | GET | Data ingestion system health |
| `/api/admin/import-data` | POST | Manual data import trigger |
| `/api/admin/data-status` | GET | Cached data status |

## 📋 Data Models

### Core Models
```python
class StockData(BaseModel):
    symbol: str
    company: str
    price: float
    change: float
    changePercent: float
    logo: Optional[str] = None

class MarketData(BaseModel):
    high: float
    low: float
    open: float
    close: float
    prevClose: float
    currentPrice: float
    symbol: str
    company: str

class NewsItem(BaseModel):
    id: str
    title: str
    summary: str
    timestamp: datetime
    source: str

class ChartDataPoint(BaseModel):
    time: str
    price: float
```

### Advanced Models
```python
class TechnicalAnalysis(BaseModel):
    symbol: str
    trend: str
    rsi: float
    macd: Dict[str, float]
    bollinger_bands: Dict[str, float]
    support_level: float
    resistance_level: float
    recommendation: str
    confidence: float

class CompanyInfo(BaseModel):
    symbol: str
    company_name: str
    sector: str
    industry: str
    market_cap: Optional[float]
    description: str
    website: Optional[str]
    employees: Optional[int]

class RiskAnalysis(BaseModel):
    symbol: str
    risk_level: str
    volatility_estimate: float
    current_price: float
    risk_metrics: Dict[str, float]
    recommendation: str
    confidence: float
    data_source: str
```

## 🔄 MCP Server Integration

The API integrates with 13 MCP (Model Context Protocol) servers:

1. **finance_market_data** - Real-time market data
2. **finance_analysis_and_predictions** - AI analysis and predictions
3. **finance_portfolio** - Portfolio management
4. **finance_news_and_insights** - News and sentiment analysis
5. **finance_calculations** - Financial calculations
6. **finance_plotting** - Chart generation
7. **finance_db_company** - Company database access
8. **finance_db_stock_price** - Stock price database access
9. **finance_data_ingestion** - Data import management
10. **finance_symbol_discovery** - Symbol search and discovery
11. **math** - Mathematical operations
12. **user_db** - User data management
13. **echo** - System testing

## 🚀 Usage Examples

### Get Market Overview
```bash
curl -X GET "http://localhost:8000/api/market-overview"
```

### Get Stock Data
```bash
curl -X GET "http://localhost:8000/api/stock/AAPL"
```

### Get Technical Analysis
```bash
curl -X GET "http://localhost:8000/api/technical-analysis/TSLA?period=6months"
```

### Get Portfolio Data
```bash
curl -X GET "http://localhost:8000/api/portfolio"
```

### Calculate Compound Return
```bash
curl -X GET "http://localhost:8000/api/calculations/compound-return?initial_amount=10000&annual_rate=0.08&years=5"
```

### Search Symbols
```bash
curl -X GET "http://localhost:8000/api/search/symbols?query=tech&limit=5"
```

## 📊 Response Format

All API responses follow a consistent structure:

```json
{
  "status": "success|error|fallback",
  "data": { /* response data */ },
  "source": "local_database|mcp_database|mcp_server|fallback",
  "timestamp": "2025-07-29T10:30:00Z"
}
```

## 🔒 Error Handling

The API implements comprehensive error handling:

- **Database Unavailable**: Falls back to MCP servers
- **MCP Server Error**: Falls back to calculated/sample data
- **Network Issues**: Returns cached data when available
- **Invalid Symbols**: Returns error with suggested alternatives

## 🚦 Health Monitoring

### System Health Endpoints
- `/api/health` - Overall API health
- `/api/health/modular` - Modular architecture status
- `/api/data-ingestion/health` - Data ingestion system status

### Health Response
```json
{
  "status": "healthy",
  "service": "modular_finance_api", 
  "version": "4.0.0",
  "modules": 3,
  "endpoints": 37,
  "mcp_servers": 13,
  "symbols_tracked": 21
}
```

## 🔧 Configuration

### Environment Variables
- `FINANCE_DB_URI` - Finance database connection string
- `USER_DB_URI` - User database connection string
- `API_KEY_*` - External API keys (when needed)

### Database Tables
- `public.stock_price` - Stock price data
- `public.company` - Company information
- `public.users` - User data

## 📝 Development Notes

### Modular Benefits
- **Maintainability**: Separated into logical modules
- **Readability**: Smaller, focused files
- **Scalability**: Easy to add new modules
- **Testing**: Individual modules can be tested separately

### Database-First Strategy
- **Performance**: Local data served instantly
- **Cost Optimization**: Minimizes external API calls
- **Reliability**: Multiple fallback layers
- **Caching**: Intelligent background refresh

## 🎯 Frontend Integration

The API is designed for seamless frontend integration:

- **TypeScript Compatible**: All models have TypeScript definitions
- **Consistent Responses**: Standardized response format
- **Error Resilience**: Never returns empty responses
- **Real-time Data**: WebSocket support planned
- **Caching Headers**: Optimized for frontend caching

## 📈 Performance

### Optimization Features
- **Database Connection Pooling**
- **Background Data Refresh**
- **Intelligent Caching Strategy**
- **Fallback Data Generation**
- **Modular Loading**

### Expected Response Times
- Local Database: < 50ms
- MCP Database: < 200ms
- MCP External: < 1000ms
- Fallback: < 100ms

---

*Last Updated: July 29, 2025*
*API Version: 4.0.0*
*Architecture: Modular*
