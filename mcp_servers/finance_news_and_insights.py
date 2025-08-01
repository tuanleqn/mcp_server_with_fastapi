import os
from mcp.server.fastmcp import FastMCP
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

load_dotenv()

# Simplified news server with reduced API dependencies
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", None)
NEWSAPI_URL = "https://newsapi.org/v2/everything"

mcp = FastMCP(name="Finance MCP Server - News (Simplified)")

@mcp.tool(description="Retrieves recent financial news with basic sentiment analysis.")
def get_financial_news(symbol: str = None, limit: int = 10) -> dict:
    """
    Retrieves financial news from NewsAPI with basic sentiment analysis.
    
    Args:
        symbol (str, optional): Stock symbol to filter news. If None, gets general market news.
        limit (int): Maximum number of articles to retrieve (default 10).
    
    Returns:
        dict: News articles with basic metadata.
    """
    try:
        if not NEWSAPI_KEY:
            # Return mock data when API key is not available
            return {
                "articles": [
                    {
                        "id": "mock_1",
                        "title": "Stock Market Update - General News",
                        "summary": "Latest market movements and financial news.",
                        "url": "https://example.com/news",
                        "timestamp": datetime.now().isoformat(),
                        "source": "Mock Financial News",
                        "sentiment": "neutral"
                    }
                ],
                "total_count": 1,
                "symbol": symbol,
                "source": "mock_data",
                "last_updated": datetime.now().isoformat()
            }
        
        # Build query parameters
        query = symbol if symbol else "financial market stocks"
        params = {
            'q': f"{query} finance",
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': min(limit, 50),
            'apiKey': NEWSAPI_KEY
        }
        
        response = requests.get(NEWSAPI_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = []
            
            for i, article in enumerate(data.get('articles', [])[:limit]):
                # Basic sentiment analysis (simplified)
                title_lower = article.get('title', '').lower()
                sentiment = 'neutral'
                if any(word in title_lower for word in ['surge', 'rise', 'gain', 'bull', 'up', 'positive']):
                    sentiment = 'positive'
                elif any(word in title_lower for word in ['fall', 'drop', 'decline', 'bear', 'down', 'negative']):
                    sentiment = 'negative'
                
                articles.append({
                    "id": f"news_{i}",
                    "title": article.get('title', 'No title'),
                    "summary": article.get('description', 'No description'),
                    "url": article.get('url', ''),
                    "timestamp": article.get('publishedAt', datetime.now().isoformat()),
                    "source": article.get('source', {}).get('name', 'Unknown'),
                    "sentiment": sentiment
                })
            
            return {
                "articles": articles,
                "total_count": len(articles),
                "symbol": symbol,
                "source": "newsapi",
                "last_updated": datetime.now().isoformat()
            }
        else:
            return {"error": f"Failed to fetch news: {response.status_code}"}
            
    except Exception as e:
        return {"error": f"News retrieval failed: {str(e)}"}

@mcp.tool(description="Analyzes basic market sentiment from recent news.")
def analyze_market_sentiment(symbol: str = None) -> dict:
    """
    Analyzes market sentiment based on recent news headlines.
    
    Args:
        symbol (str, optional): Stock symbol to analyze sentiment for.
    
    Returns:
        dict: Basic sentiment analysis results.
    """
    try:
        # Get recent news
        news_data = get_financial_news(symbol=symbol, limit=20)
        
        if "error" in news_data:
            return news_data
        
        articles = news_data.get("articles", [])
        if not articles:
            return {
                "symbol": symbol,
                "overall_sentiment": "neutral",
                "sentiment_score": 0,
                "confidence": 0,
                "article_count": 0,
                "analysis_date": datetime.now().isoformat()
            }
        
        # Simple sentiment scoring
        sentiment_scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            sentiment = article.get("sentiment", "neutral")
            if sentiment == "positive":
                sentiment_scores.append(1)
                positive_count += 1
            elif sentiment == "negative":
                sentiment_scores.append(-1)
                negative_count += 1
            else:
                sentiment_scores.append(0)
                neutral_count += 1
        
        # Calculate overall sentiment
        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            overall_sentiment = "positive" if avg_score > 0.2 else "negative" if avg_score < -0.2 else "neutral"
            confidence = min(100, abs(avg_score) * 100)
        else:
            avg_score = 0
            overall_sentiment = "neutral"
            confidence = 0
        
        return {
            "symbol": symbol,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(avg_score * 100, 2),  # Scale to -100 to 100
            "confidence": round(confidence, 2),
            "sentiment_breakdown": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count
            },
            "article_count": len(articles),
            "analysis_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {str(e)}"}

# Simplified news server with only 2 essential tools
# Removed: get_breaking_news, get_market_news (wrapper) functions

if __name__ == "__main__":
    mcp.run()
