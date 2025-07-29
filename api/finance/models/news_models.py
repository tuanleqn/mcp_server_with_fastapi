"""
News-related Pydantic models
"""

from pydantic import BaseModel
from datetime import datetime

class NewsItem(BaseModel):
    id: str
    title: str
    summary: str
    timestamp: datetime
    source: str
