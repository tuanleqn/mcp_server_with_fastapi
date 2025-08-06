"""
Finance News and Insights Server
Simplified MCP server for financial news and sentiment analysis
"""

import os
import requests
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWSAPI_KEY")  # Fixed: using NEWSAPI_KEY from .env

mcp = FastMCP(name="Finance News and Insights Server")


def analyze_sentiment_simple(text: str) -> dict:
    """
    Simple rule-based sentiment analysis for financial text.
    """
    if not text:
        return {"sentiment": "neutral", "score": 0, "confidence": 20}
    
    text = text.lower()
    
    # Financial positive keywords
    positive_words = [
        'gain', 'gains', 'up', 'rise', 'rises', 'rising', 'boost', 'surge', 'rally',
        'bullish', 'growth', 'profit', 'profits', 'earnings', 'beat', 'strong',
        'outperform', 'buy', 'upgrade', 'positive', 'optimistic', 'recover', 'recovery'
    ]
    
    # Financial negative keywords  
    negative_words = [
        'fall', 'falls', 'falling', 'drop', 'drops', 'decline', 'down', 'loss',
        'losses', 'bearish', 'crash', 'plunge', 'sell', 'downgrade', 'negative',
        'pessimistic', 'concern', 'concerns', 'risk', 'risks', 'weak', 'underperform'
    ]
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    total_count = positive_count + negative_count
    if total_count == 0:
        return {"sentiment": "neutral", "score": 0, "confidence": 20}
    
    sentiment_score = (positive_count - negative_count) / total_count
    
    if sentiment_score > 0.3:
        sentiment = "positive"
    elif sentiment_score < -0.3:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    confidence = min(80, max(20, abs(sentiment_score) * 100 + total_count * 10))
    
    return {
        "sentiment": sentiment,
        "score": round(sentiment_score, 2),
        "confidence": round(confidence)
    }


@mcp.tool(description="Get financial news headlines")
def get_financial_news(query: str = "financial markets", limit: int = 10) -> dict:
    """
    Get financial news headlines using NewsAPI.
    
    Args:
        query: Search query for news (default: "financial markets")
        limit: Maximum number of articles (default: 10, max: 50)
    
    Returns:
        Dictionary containing news articles with sentiment analysis
    """
    limit = min(limit, 50)
    
    print(f"DEBUG: NEWS_API_KEY configured: {bool(NEWS_API_KEY)}")
    print(f"DEBUG: NEWS_API_KEY length: {len(NEWS_API_KEY) if NEWS_API_KEY else 0}")
    print(f"DEBUG: NEWS_API_KEY value: {NEWS_API_KEY[:10]}..." if NEWS_API_KEY else "DEBUG: NEWS_API_KEY is None/empty")
    
    if not NEWS_API_KEY or NEWS_API_KEY.strip() == "":
        return {
            "success": False,
            "error": "NEWS_API_KEY not configured. Please add NEWSAPI_KEY to your .env file to get real financial news",
            "query": query,
            "articles_found": 0,
            "articles": []
        }
    
    try:
        # Use NewsAPI everything endpoint for better results
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{query} finance OR stock OR market",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": NEWS_API_KEY.strip()
        }
        
        print(f"DEBUG: Making NewsAPI request to: {url}")
        print(f"DEBUG: Query: {params['q']}")
        
        response = requests.get(url, params=params, timeout=10)
        print(f"DEBUG: NewsAPI response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"DEBUG: NewsAPI returned {len(data.get('articles', []))} articles")
            print(f"DEBUG: Raw API response keys: {list(data.keys())}")
            if data.get('articles'):
                print(f"DEBUG: First article title: {data['articles'][0].get('title', 'NO_TITLE')}")
                print(f"DEBUG: First article source: {data['articles'][0].get('source', {}).get('name', 'NO_SOURCE')}")
            
            articles = []
            
            for article in data.get("articles", []):
                if article.get("title") and article.get("description"):
                    # Analyze sentiment for each article
                    text_for_analysis = f"{article['title']} {article['description']}"
                    sentiment = analyze_sentiment_simple(text_for_analysis)
                    
                    articles.append({
                        "title": article["title"],
                        "description": article["description"],
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "sentiment": sentiment
                    })
            
            print(f"DEBUG: Processed {len(articles)} articles")
            if articles:
                print(f"DEBUG: Final first article title: {articles[0]['title']}")
                print(f"DEBUG: Final first article source: {articles[0]['source']}")
            
            return {
                "success": True,
                "query": query,
                "articles_found": len(articles),
                "articles": articles,
                "retrieved_at": datetime.now().isoformat(),
                "source": "newsapi"
            }
        else:
            error_msg = f"NewsAPI error: {response.status_code}"
            try:
                error_detail = response.json().get("message", "")
                if error_detail:
                    error_msg += f" - {error_detail}"
            except:
                pass
            
            print(f"DEBUG: NewsAPI error: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "query": query,
                "articles_found": 0,
                "articles": []
            }
            
    except Exception as e:
        print(f"DEBUG: NewsAPI exception: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to fetch news: {str(e)}",
            "query": query,
            "articles_found": 0,
            "articles": []
        }


@mcp.tool(description="Analyze market sentiment from news")
def get_market_sentiment(query: str = "stock market", limit: int = 20) -> dict:
    """
    Analyze overall market sentiment based on recent news.
    
    Args:
        query: Search query for sentiment analysis (default: "stock market")
        limit: Number of articles to analyze (default: 20)
    
    Returns:
        Dictionary containing sentiment analysis results
    """
    # Get news articles
    news_result = get_financial_news(query, limit)
    
    if not news_result.get("success"):
        return news_result
    
    articles = news_result.get("articles", [])
    if not articles:
        return {
            "success": False,
            "error": "No articles found for sentiment analysis",
            "query": query
        }
    
    # Aggregate sentiment scores
    sentiments = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for article in articles:
        sentiment_data = article.get("sentiment", {})
        sentiment = sentiment_data.get("sentiment", "neutral")
        score = sentiment_data.get("score", 0)
        
        sentiments.append(score)
        
        if sentiment == "positive":
            positive_count += 1
        elif sentiment == "negative":
            negative_count += 1
        else:
            neutral_count += 1
    
    # Calculate overall sentiment
    if sentiments:
        avg_score = sum(sentiments) / len(sentiments)
        if avg_score > 0.2:
            overall_sentiment = "positive"
        elif avg_score < -0.2:
            overall_sentiment = "negative"  
        else:
            overall_sentiment = "neutral"
    else:
        avg_score = 0
        overall_sentiment = "neutral"
    
    # Calculate confidence based on consensus
    total_articles = len(articles)
    max_category = max(positive_count, negative_count, neutral_count)
    confidence = (max_category / total_articles * 100) if total_articles > 0 else 0
    
    return {
        "success": True,
        "query": query,
        "overall_sentiment": overall_sentiment,
        "sentiment_score": round(avg_score, 3),
        "confidence": round(confidence, 1),
        "sentiment_breakdown": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "total_articles": total_articles
        },
        "market_indicators": {
            "bullish_signals": positive_count,
            "bearish_signals": negative_count,
            "neutral_signals": neutral_count,
            "sentiment_strength": "strong" if confidence > 60 else "moderate" if confidence > 40 else "weak"
        },
        "analysis_date": datetime.now().isoformat()
    }


if __name__ == "__main__":
    mcp.run()
