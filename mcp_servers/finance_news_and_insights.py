import os
from mcp.server.fastmcp import FastMCP
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

load_dotenv()

# Multiple API configurations for redundancy
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", None)  # NewsAPI.org
ALPHA_VANTAGE_KEY = os.getenv("EXTERNAL_FINANCE_API_KEY", None)  # Alpha Vantage
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", None)  # Finnhub

# API endpoints
NEWSAPI_URL = "https://newsapi.org/v2/everything"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
FINNHUB_URL = "https://finnhub.io/api/v1"

mcp = FastMCP(name="Finance MCP Server - News and Insights")

@mcp.tool(description="Retrieves recent financial news from multiple sources with sentiment analysis.")
def get_financial_news(symbol: str = None, limit: int = 10, source: str = "auto") -> dict:
    """
    Retrieves financial news from multiple sources with fallback options.
    
    Args:
        symbol (str, optional): Stock symbol to filter news. If None, gets general market news.
        limit (int): Maximum number of articles to retrieve (default 10).
        source (str): News source - 'auto', 'newsapi', 'alpha_vantage', or 'finnhub'.
    
    Returns:
        dict: News articles with metadata and sentiment analysis.
    """
    news_data = []
    
    # Try different sources based on preference and availability
    sources_to_try = []
    if source == "auto":
        if NEWSAPI_KEY:
            sources_to_try.append("newsapi")
        if ALPHA_VANTAGE_KEY:
            sources_to_try.append("alpha_vantage")
        if FINNHUB_KEY:
            sources_to_try.append("finnhub")
    else:
        sources_to_try.append(source)
    
    if not sources_to_try:
        return {"error": "No API keys configured. Please set NEWSAPI_KEY, EXTERNAL_FINANCE_API_KEY, or FINNHUB_API_KEY in environment."}
    
    for source_name in sources_to_try:
        try:
            if source_name == "newsapi" and NEWSAPI_KEY:
                news_data = _get_news_from_newsapi(symbol, limit)
                if news_data:
                    break
            elif source_name == "alpha_vantage" and ALPHA_VANTAGE_KEY:
                news_data = _get_news_from_alpha_vantage(symbol, limit)
                if news_data:
                    break
            elif source_name == "finnhub" and FINNHUB_KEY:
                news_data = _get_news_from_finnhub(symbol, limit)
                if news_data:
                    break
        except Exception as e:
            print(f"Error with {source_name}: {e}")
            continue
    
    if not news_data:
        return {"error": "Failed to retrieve news from all available sources."}
    
    return {
        "success": True,
        "symbol": symbol,
        "total_articles": len(news_data),
        "articles": news_data[:limit],
        "source_used": source if source != "auto" else "multiple",
        "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def _get_news_from_newsapi(symbol: str = None, limit: int = 10) -> list:
    """Get news from NewsAPI.org with improved search parameters"""
    try:
        # Build more flexible query
        if symbol:
            query = f'("{symbol}" OR "{symbol} stock" OR "{symbol} shares") AND (finance OR stock OR market OR earnings OR trading)'
        else:
            query = "(stock market OR financial news OR economy OR trading OR investment OR earnings OR finance)"
        
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": NEWSAPI_KEY,
            # Remove strict domain restrictions - use broader sources
            "sources": "bloomberg,reuters,cnbc,the-wall-street-journal,financial-times,yahoo-news,associated-press,cnn,bbc-news"
        }
        
        response = requests.get(NEWSAPI_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            # If sources fail, try without source restriction
            print(f"NewsAPI: Trying without source restrictions...")
            params.pop("sources", None)
            params["from"] = "2024-12-01"  # Get recent articles
            
            response = requests.get(NEWSAPI_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        
        if data.get("status") != "ok" or not data.get("articles"):
            return []
        
        articles = []
        for article in data.get("articles", [])[:limit]:
            # Skip articles with missing essential data
            if not article.get("title") or article.get("title") == "[Removed]":
                continue
                
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", "NewsAPI"),
                "published_at": article.get("publishedAt", ""),
                "author": article.get("author", "Unknown"),
                "sentiment": _analyze_sentiment(article.get("title", "") + " " + article.get("description", ""))
            })
        
        return articles
    
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


def _get_news_from_alpha_vantage(symbol: str = None, limit: int = 10) -> list:
    """Get news from Alpha Vantage NEWS_SENTIMENT endpoint"""
    try:
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": ALPHA_VANTAGE_KEY,
            "limit": limit
        }
        
        if symbol:
            params["tickers"] = symbol.upper()
        
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "feed" not in data:
            return []
        
        articles = []
        for item in data["feed"][:limit]:
            # Extract sentiment if available
            sentiment = "neutral"
            if "overall_sentiment_score" in item:
                score = float(item["overall_sentiment_score"])
                if score > 0.1:
                    sentiment = "positive"
                elif score < -0.1:
                    sentiment = "negative"
            
            articles.append({
                "title": item.get("title", ""),
                "description": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", "Alpha Vantage"),
                "published_at": item.get("time_published", ""),
                "author": item.get("authors", ["Unknown"])[0] if item.get("authors") else "Unknown",
                "sentiment": sentiment,
                "sentiment_score": item.get("overall_sentiment_score", 0),
                "relevance_score": item.get("relevance_score", 0) if symbol else None
            })
        
        return articles
    
    except Exception as e:
        print(f"Alpha Vantage error: {e}")
        return []


def _get_news_from_finnhub(symbol: str = None, limit: int = 10) -> list:
    """Get news from Finnhub API with proper date formatting"""
    try:
        if symbol:
            # Company news with proper date format (YYYY-MM-DD)
            from datetime import datetime, timedelta
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            params = {
                "symbol": symbol.upper(),
                "from": from_date,
                "to": to_date,
                "token": FINNHUB_KEY
            }
            endpoint = f"{FINNHUB_URL}/company-news"
        else:
            # General market news (this endpoint works better)
            params = {
                "category": "general",
                "token": FINNHUB_KEY
            }
            endpoint = f"{FINNHUB_URL}/news"
        
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Handle error responses
        if isinstance(data, dict) and "error" in data:
            print(f"Finnhub API error: {data['error']}")
            return []
        
        if not isinstance(data, list):
            print(f"Finnhub unexpected response format: {type(data)}")
            return []
        
        articles = []
        for item in data[:limit]:
            # Skip items without essential data
            headline = item.get("headline", "")
            if not headline or headline.strip() == "":
                continue
                
            # Convert timestamp to ISO format
            published_at = ""
            if item.get("datetime"):
                try:
                    published_at = datetime.fromtimestamp(item["datetime"]).isoformat()
                except (ValueError, TypeError):
                    published_at = ""
            
            articles.append({
                "title": headline,
                "description": item.get("summary", "")[:200] + "..." if len(item.get("summary", "")) > 200 else item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", "Finnhub"),
                "published_at": published_at,
                "author": "Finnhub",
                "sentiment": _analyze_sentiment(headline + " " + item.get("summary", ""))
            })
        
        return articles
    
    except Exception as e:
        print(f"Finnhub error: {e}")
        # Fallback to general news if company news fails
        if symbol:
            try:
                params = {
                    "category": "general", 
                    "token": FINNHUB_KEY
                }
                response = requests.get(f"{FINNHUB_URL}/news", params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        return [{"title": item.get("headline", ""), 
                                "description": item.get("summary", ""),
                                "url": item.get("url", ""),
                                "source": "Finnhub", 
                                "published_at": datetime.fromtimestamp(item.get("datetime", 0)).isoformat() if item.get("datetime") else "",
                                "author": "Finnhub",
                                "sentiment": _analyze_sentiment(item.get("headline", ""))} 
                               for item in data[:limit] if item.get("headline")]
            except:
                pass
        return []


def _analyze_sentiment(text: str) -> str:
    """Simple sentiment analysis based on keywords"""
    if not text:
        return "neutral"
    
    text_lower = text.lower()
    
    positive_words = [
        "gain", "rise", "surge", "jump", "soar", "climb", "rally", "boost", "increase",
        "profit", "earnings", "beat", "exceed", "strong", "growth", "bullish", "buy",
        "upgrade", "positive", "optimistic", "recover", "breakthrough", "success"
    ]
    
    negative_words = [
        "fall", "drop", "decline", "plunge", "crash", "sink", "tumble", "slide", "loss",
        "deficit", "miss", "disappoint", "weak", "bearish", "sell", "downgrade",
        "negative", "pessimistic", "concern", "risk", "crisis", "uncertainty", "fear"
    ]
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"


@mcp.tool(description="Analyzes market sentiment from recent news articles and provides insights.")
def analyze_market_sentiment(symbol: str = None, days_back: int = 7) -> dict:
    """
    Analyzes market sentiment from recent news articles.
    
    Args:
        symbol (str, optional): Stock symbol to analyze. If None, analyzes general market sentiment.
        days_back (int): Number of days to look back for news (default 7).
    
    Returns:
        dict: Sentiment analysis with insights and recommendations.
    """
    # Get recent news
    news_result = get_financial_news(symbol=symbol, limit=20)
    
    if not news_result.get("success"):
        return {"error": "Failed to retrieve news for sentiment analysis."}
    
    articles = news_result.get("articles", [])
    
    if not articles:
        return {"error": "No articles found for sentiment analysis."}
    
    # Analyze sentiment distribution
    sentiments = [article.get("sentiment", "neutral") for article in articles]
    positive_count = sentiments.count("positive")
    negative_count = sentiments.count("negative")
    neutral_count = sentiments.count("neutral")
    total_count = len(sentiments)
    
    # Calculate percentages
    positive_pct = (positive_count / total_count) * 100
    negative_pct = (negative_count / total_count) * 100
    neutral_pct = (neutral_count / total_count) * 100
    
    # Determine overall sentiment
    if positive_pct > negative_pct + 10:
        overall_sentiment = "bullish"
        confidence = min(positive_pct / 60 * 100, 100)  # Scale confidence
    elif negative_pct > positive_pct + 10:
        overall_sentiment = "bearish"
        confidence = min(negative_pct / 60 * 100, 100)
    else:
        overall_sentiment = "neutral"
        confidence = max(neutral_pct, 50)
    
    # Generate insights
    insights = []
    if positive_pct > 60:
        insights.append("Strong positive sentiment indicates potential bullish momentum.")
    elif negative_pct > 60:
        insights.append("Strong negative sentiment suggests caution and potential downside risk.")
    elif neutral_pct > 60:
        insights.append("Neutral sentiment indicates market uncertainty or lack of clear direction.")
    
    if positive_pct > 0 and negative_pct > 0:
        insights.append(f"Mixed sentiment with {positive_pct:.1f}% positive vs {negative_pct:.1f}% negative news.")
    
    # Extract key themes from titles
    all_titles = " ".join([article.get("title", "") for article in articles])
    common_themes = _extract_themes(all_titles)
    
    return {
        "success": True,
        "symbol": symbol or "market",
        "analysis_period": f"{days_back} days",
        "total_articles_analyzed": total_count,
        "overall_sentiment": overall_sentiment,
        "confidence_score": round(confidence, 1),
        "sentiment_distribution": {
            "positive": {
                "count": positive_count,
                "percentage": round(positive_pct, 1)
            },
            "negative": {
                "count": negative_count,
                "percentage": round(negative_pct, 1)
            },
            "neutral": {
                "count": neutral_count,
                "percentage": round(neutral_pct, 1)
            }
        },
        "insights": insights,
        "key_themes": common_themes,
        "recommendation": _generate_sentiment_recommendation(overall_sentiment, confidence),
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def _extract_themes(text: str) -> list:
    """Extract common themes from news titles"""
    if not text:
        return []
    
    text_lower = text.lower()
    
    themes = {
        "earnings": ["earnings", "revenue", "profit", "loss", "quarterly", "annual"],
        "mergers_acquisitions": ["merger", "acquisition", "buyout", "takeover", "deal"],
        "regulation": ["regulation", "regulatory", "sec", "fda", "government", "policy"],
        "technology": ["technology", "tech", "ai", "digital", "innovation", "software"],
        "market_trends": ["market", "trend", "outlook", "forecast", "prediction"],
        "economic_indicators": ["inflation", "gdp", "employment", "interest", "fed", "economy"],
        "crypto": ["bitcoin", "crypto", "blockchain", "digital currency"],
        "energy": ["oil", "gas", "energy", "renewable", "solar", "wind"],
        "healthcare": ["healthcare", "pharma", "drug", "medical", "health"]
    }
    
    found_themes = []
    for theme, keywords in themes.items():
        if any(keyword in text_lower for keyword in keywords):
            found_themes.append(theme.replace("_", " ").title())
    
    return found_themes[:5]  # Return top 5 themes


def _generate_sentiment_recommendation(sentiment: str, confidence: float) -> str:
    """Generate trading recommendation based on sentiment"""
    if sentiment == "bullish" and confidence > 70:
        return "Consider long positions with proper risk management. Strong positive sentiment detected."
    elif sentiment == "bearish" and confidence > 70:
        return "Exercise caution. Consider defensive positions or wait for better entry points."
    elif sentiment == "neutral" or confidence < 50:
        return "Market sentiment is unclear. Wait for more definitive signals before making major moves."
    else:
        return "Moderate sentiment detected. Consider small position sizes and close monitoring."


@mcp.tool(description="Gets breaking financial news and market alerts.")
def get_breaking_news(limit: int = 5) -> dict:
    """
    Retrieves the most recent breaking financial news and market alerts.
    
    Args:
        limit (int): Maximum number of breaking news items to retrieve.
    
    Returns:
        dict: Most recent breaking news with urgency indicators.   
    """
    # Get very recent news (last few hours)
    news_result = get_financial_news(limit=limit * 2)  # Get more to filter
    
    if not news_result.get("success"):
        return {"error": "Failed to retrieve breaking news."}
    
    articles = news_result.get("articles", [])
    
    # Filter for very recent articles (last 6 hours)
    recent_cutoff = datetime.now() - timedelta(hours=6)
    breaking_news = []
    
    for article in articles:
        published_str = article.get("published_at", "")
        if published_str:
            try:
                # Parse different datetime formats
                if "T" in published_str:
                    published_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                else:
                    published_date = datetime.strptime(published_str, "%Y-%m-%d %H:%M:%S")
                
                if published_date.replace(tzinfo=None) > recent_cutoff:
                    # Add urgency score based on keywords
                    title = article.get("title", "").lower()
                    urgency = "normal"
                    
                    urgent_keywords = ["breaking", "alert", "crash", "surge", "halt", "emergency"]
                    high_keywords = ["major", "significant", "important", "critical", "urgent"]
                    
                    if any(keyword in title for keyword in urgent_keywords):
                        urgency = "urgent"
                    elif any(keyword in title for keyword in high_keywords):
                        urgency = "high"
                    
                    article["urgency"] = urgency
                    breaking_news.append(article)
                    
            except (ValueError, TypeError):
                continue  # Skip articles with unparseable dates
    
    # Sort by urgency and limit results
    urgency_order = {"urgent": 0, "high": 1, "normal": 2}
    breaking_news.sort(key=lambda x: urgency_order.get(x.get("urgency", "normal"), 2))
    
    return {
        "success": True,
        "breaking_news_count": len(breaking_news[:limit]),
        "articles": breaking_news[:limit],
        "alert_level": "high" if any(a.get("urgency") == "urgent" for a in breaking_news[:limit]) else "normal",
        "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }