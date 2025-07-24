# Finance MCP Server with FastAPI

A comprehensive FastAPI application that integrates multiple Model Context Protocol (MCP) servers for financial analysis, data management, and computational tools.

## Project Overview

This project implements a production-ready FastAPI application that mounts multiple specialized MCP servers:

### Core Tools
- **Echo Server**: Simple echo functionality for testing
- **Math Server**: Mathematical operations and calculations
- **User Database Server**: User authentication and data management

### Finance Tools
- **Company Database**: Company information and financial data
- **Stock Price Database**: Real-time and historical stock price data
- **Data Ingestion**: Financial data import and processing
- **Financial Calculations**: Advanced financial computations
- **Portfolio Management**: Investment portfolio tracking and analysis
- **Data Visualization**: Financial charts and plotting capabilities
- **News & Insights**: Financial news aggregation and analysis
- **Analysis & Predictions**: Financial forecasting and predictive analytics

## Prerequisites

- Python 3.12 or higher
- PostgreSQL database (for finance and user data)
- UV package manager for dependency management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multimcp-server-with-fastapi
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv sync
```

## Database Setup

1. Create a PostgreSQL database for the application
2. Run the database setup script:
```bash
psql -d your_database -f database_setup.sql
```

This will create the necessary tables for user authentication and sample data.

## Configuration

Create a `.env` file in the root directory with the following variables:
```env
HOST=localhost
PORT=8000
LOG_LV=debug
DB_URI=postgresql://username:password@localhost:5432/your_finance_db
```

Replace `username`, `password`, and `your_finance_db` with your actual PostgreSQL credentials and database name.

## Running the Server

Start the server using:
```bash
python main.py
```

The server will start on `http://127.0.0.1:8000` with auto-reload enabled for development.

## API Endpoints

### Core Endpoints
- `GET /` - Root endpoint with server information
- `GET /health` - Health check and server status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /static/*` - Static file serving

### MCP Tool Endpoints
- `/echo/` - Echo server tools
- `/math/` - Mathematical operations
- `/user_db/` - User database operations
- `/finance_db_company/` - Company data queries
- `/finance_db_stock_price/` - Stock price data
- `/finance_data_ingestion/` - Data import tools
- `/finance_calculations/` - Financial calculations
- `/finance_portfolio/` - Portfolio management
- `/finance_plotting/` - Data visualization
- `/finance_news_and_insights/` - News and market insights
- `/finance_analysis_and_predictions/` - Predictive analytics

## Development

The server includes auto-reload functionality for development. Access the interactive API documentation at `http://127.0.0.1:8000/docs` to explore and test all available endpoints.

## Project Structure

```
├── main.py                 # FastAPI application entry point
├── database_setup.sql      # Database schema and sample data
├── mcp_servers/           # Individual MCP server implementations
│   ├── echo.py
│   ├── math.py
│   ├── user_db.py
│   └── finance_*.py       # Finance-specific MCP servers
├── utils/                 # Utility modules
│   └── auth.py           # Authentication utilities
├── static/               # Static files (if needed)
├── pyproject.toml        # Project configuration
└── .env                  # Environment variables (create this)
```
