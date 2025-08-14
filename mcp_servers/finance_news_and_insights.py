"""
Finance News and Insights Server
"""

import os
import requests
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWSAPI_KEY")

mcp = FastMCP(name="Finance News and Insights Server")


def analyze_sentiment_simple(text: str) -> dict:
    if not text:
        return {"sentiment": "neutral", "score": 0, "confidence": 20}

    text = text.lower()

    # Financial positive keywords
    positive_words = [
        "gain",
        "gains",
        "up",
        "rise",
        "rises",
        "rising",
        "boost",
        "surge",
        "rally",
        "bullish",
        "growth",
        "profit",
        "profits",
        "earnings",
        "beat",
        "strong",
        "outperform",
        "buy",
        "upgrade",
        "positive",
        "optimistic",
        "recover",
        "recovery",
    ]

    # Financial negative keywords
    negative_words = [
        "fall",
        "falls",
        "falling",
        "drop",
        "drops",
        "decline",
        "down",
        "loss",
        "losses",
        "bearish",
        "crash",
        "plunge",
        "sell",
        "downgrade",
        "negative",
        "pessimistic",
        "concern",
        "concerns",
        "risk",
        "risks",
        "weak",
        "underperform",
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
        "confidence": round(confidence),
    }


@mcp.tool(description="Get financial news headlines")
def get_financial_news(query: str = "financial markets", limit: int = 10) -> dict:
    limit = min(limit, 50)

    if not NEWS_API_KEY or NEWS_API_KEY.strip() == "":
        return {
            "success": False,
            "error": "NEWS_API_KEY not configured",
            "query": query,
            "articles_found": 0,
            "articles": [],
        }

    try:
        # Use NewsAPI everything endpoint for better results
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{query} finance OR stock OR market",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": NEWS_API_KEY.strip(),
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            articles = []

            for article in data.get("articles", []):
                if article.get("title") and article.get("description"):
                    text_for_analysis = f"{article['title']} {article['description']}"
                    sentiment = analyze_sentiment_simple(text_for_analysis)

                    articles.append(
                        {
                            "title": article["title"],
                            "description": article["description"],
                            "url": article.get("url", ""),
                            "published_at": article.get("publishedAt", ""),
                            "source": article.get("source", {}).get("name", "Unknown"),
                            "sentiment": sentiment,
                        }
                    )

            return {
                "success": True,
                "query": query,
                "articles_found": len(articles),
                "articles": articles,
                "retrieved_at": datetime.now().isoformat(),
                "source": "newsapi",
            }
        else:
            error_msg = f"NewsAPI error: {response.status_code}"
            try:
                error_detail = response.json().get("message", "")
                if error_detail:
                    error_msg += f" - {error_detail}"
            except:
                pass

            return {
                "success": False,
                "error": error_msg,
                "query": query,
                "articles_found": 0,
                "articles": [],
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to fetch news: {str(e)}",
            "query": query,
            "articles_found": 0,
            "articles": [],
        }


@mcp.tool(description="Analyze market sentiment from news")
def get_market_sentiment(query: str = "stock market", limit: int = 20) -> dict:
    news_result = get_financial_news(query, limit)

    if not news_result.get("success"):
        return news_result

    articles = news_result.get("articles", [])
    if not articles:
        return {
            "success": False,
            "error": "No articles found for sentiment analysis",
            "query": query,
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
            "total_articles": total_articles,
        },
        "market_indicators": {
            "bullish_signals": positive_count,
            "bearish_signals": negative_count,
            "neutral_signals": neutral_count,
            "sentiment_strength": "strong"
            if confidence > 60
            else "moderate"
            if confidence > 40
            else "weak",
        },
        "news_articles": [
            {
                "title": article.get("title", ""),
                "published_date": article.get("published_date", ""),
                "source": article.get("source", ""),
                "sentiment": article.get("sentiment", {}).get("sentiment", "neutral"),
                "sentiment_score": article.get("sentiment", {}).get("score", 0),
                "url": article.get("url", ""),
            }
            for article in articles
        ],
        "analysis_date": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    mcp.run()
