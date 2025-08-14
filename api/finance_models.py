"""
Finance API Data Models
Pydantic models for TypeScript-compatible API responses
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# Core Stock and Market Data Models
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

class ChartDataPoint(BaseModel):
    time: str
    price: float

class NewsItem(BaseModel):
    id: str
    title: str
    summary: str
    timestamp: datetime
    source: str

# Advanced Analysis Models
class TechnicalAnalysis(BaseModel):
    symbol: str
    trend: str  # "bullish", "bearish", "neutral"
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

# Asset Class Models
class CryptoData(BaseModel):
    symbol: str
    name: str
    price: float
    change_24h: float
    change_percent_24h: float
    market_cap: Optional[float]
    volume_24h: Optional[float]

class CommodityData(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    unit: str  # e.g., "USD per ounce", "USD per barrel"

class IndexData(BaseModel):
    symbol: str
    name: str
    value: float
    change: float
    change_percent: float
    constituents: Optional[int]

# Chart and Visualization Models
class ChartImage(BaseModel):
    symbol: str
    chart_type: str
    image_base64: str
    timestamp: datetime

# Portfolio Models
class PortfolioPosition(BaseModel):
    symbol: str
    company: str
    shares: int
    avg_cost: float
    current_price: float
    market_value: float
    gain_loss: float
    gain_loss_percent: float

class PortfolioData(BaseModel):
    total_value: float
    total_cost: float
    total_gain: float
    total_return_percent: float
    positions: List[PortfolioPosition]

# Analysis and Calculation Models
class RiskAnalysis(BaseModel):
    symbol: str
    risk_level: str
    volatility_estimate: float
    current_price: float
    risk_metrics: Dict[str, float]
    recommendation: str
    confidence: float
    data_source: str

class CompoundReturnResult(BaseModel):
    calculation: str
    inputs: Dict[str, Any]
    results: Dict[str, float]
    source: str

class CorrelationMatrix(BaseModel):
    symbols: List[str]
    correlation_matrix: Dict[str, Dict[str, float]]
    analysis: Dict[str, float]
    recommendations: List[str]

# Market Overview Models
class MarketOverview(BaseModel):
    market_sentiment: str
    average_change: float
    tech_stocks: List[Dict[str, Any]]
    major_indices: List[Dict[str, Any]]
    commodities_summary: Dict[str, str]
    last_updated: str

class MarketStatus(BaseModel):
    market_status: str
    trading_session: str
    current_time: Optional[str] = None
    market_open_time: Optional[str] = None
    market_close_time: Optional[str] = None
    timezone: str
    source: str

# User and Database Models
class UserData(BaseModel):
    status: str
    user: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    source: Optional[str] = None

class DatabaseResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    symbol: Optional[str] = None
    source: Optional[str] = None

# System Health Models
class APIHealth(BaseModel):
    status: str
    service: str
    version: str
    data_source: str
    symbol_focus: str
    features: List[str]
    core_endpoints: List[str]
    asset_classes: Dict[str, List[str]]
    symbols_tracked: int
    last_updated: str

class DataIngestionHealth(BaseModel):
    status: str
    ingestion_active: bool
    last_update: Optional[str] = None
    sources_active: Optional[List[str]] = None
    error_rate: Optional[float] = None
    message: Optional[str] = None
    fallback_mode: Optional[bool] = None
