"""
Stock-related Pydantic models
"""

from pydantic import BaseModel
from typing import Dict

class StockData(BaseModel):
    symbol: str
    company: str
    price: float
    change: float
    changePercent: float
    logo: str = None

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
