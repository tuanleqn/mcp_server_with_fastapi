import os
from mcp.server.fastmcp import FastMCP
import requests
from dotenv import load_dotenv

load_dotenv()

# Example API key. Replace with a real API service (e.g., Alpha Vantage, Finnhub, NewsAPI.org)
# Ensure you sign up for an API key and follow their terms of service.
EXTERNAL_FINANCE_API_KEY = os.getenv("EXTERNAL_FINANCE_API_KEY", None)
EXTERNAL_FINANCE_NEWS_API_BASE_URL = os.getenv(
    "EXTERNAL_FINANCE_NEWS_API_BASE_URL", "https://www.alphavantage.co/query"
)  # Example Alpha Vantage URL

if not EXTERNAL_FINANCE_API_KEY:
    raise ValueError(
        "EXTERNAL_FINANCE_API_KEY not found in environment variables. Please get an API key from a financial data provider."
    )

mcp = FastMCP(name="Finance MCP Server - News and Insights")


@mcp.tool(
    description="Retrieves recent financial news for a specific company or general market news."
)
def get_financial_news(symbol: str = None, limit: int = 5) -> dict:
    """
    Retrieves recent financial news articles. Can be filtered by a specific stock symbol.

    Args:
        symbol (str, optional): The stock symbol to fetch news for. If None, fetches general market news.
        limit (int): The maximum number of news articles to retrieve.

    Returns:
        dict: A dictionary containing a list of news articles.
              Each article includes title, URL, source, and published date.
              Returns an error if the API call fails or no news is found.
    """
    # This implementation is a placeholder for Alpha Vantage's NEWS_SENTIMENT endpoint.
    # You would need to check their specific documentation for parameters and response format.
    # Other APIs like NewsAPI.org or Finnhub have different endpoints and parameters.

    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": EXTERNAL_FINANCE_API_KEY,
        "topics": "financial_markets",  # General topic
    }
    if symbol:
        params["tickers"] = symbol.upper()  # Filter by ticker if provided

    try:
        response = requests.get(EXTERNAL_FINANCE_NEWS_API_BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        if "feed" not in data or not data["feed"]:
            return {
                "error": f"No news found for {symbol if symbol else 'general market'}."
            }

        news_articles = []
        for article in data["feed"][:limit]:
            news_articles.append(
                {
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "source": article.get("source"),
                    "published_date": article.get(
                        "time_published"
                    ),  # Alpha Vantage uses time_published
                }
            )
        return {"articles": news_articles}

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch news from external API: {str(e)}"}
    except Exception as e:
        return {"error": f"Error processing news data: {str(e)}"}


# You could add a sentiment analysis tool here if the external API provides it,
# or integrate with an NLP library (like NLTK, spaCy, or huggingface transformers)
# to analyze the sentiment of fetched news article texts.
# @mcp.tool(description="Analyzes the sentiment of recent news for a stock.")
# def analyze_news_sentiment(symbol: str) -> dict:
#    # Implementation would involve fetching news using get_financial_news
#    # then passing article text through a sentiment analysis model.
#    pass
