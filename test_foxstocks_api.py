#!/usr/bin/env python3
"""
Test script for Direct Finance API (No Authentication Required)
"""

import requests
import json
import time

def test_direct_finance_api():
    """Test the direct finance API endpoints"""
    base_url = "http://127.0.0.1:8000"
    
    print("💰 Testing Direct Finance API")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing API health check...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ API health check passed")
            print(f"Available endpoints: {len(result['endpoints'])}")
        else:
            print(f"❌ API health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API health check error: {e}")
    
    # Test 2: Market data (VN-INDEX)
    print("\n2. Testing market data (VN-INDEX)...")
    try:
        response = requests.get(f"{base_url}/api/market-data/VN-INDEX")
        if response.status_code == 200:
            result = response.json()
            print("✅ Market data successful")
            print(f"Symbol: {result['symbol']}")
            print(f"Current Price: {result['currentPrice']}")
            print(f"Change: {result['close'] - result['prevClose']:.2f}")
        else:
            print(f"❌ Market data failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Market data error: {e}")
    
    # Test 3: Watchlist stocks
    print("\n3. Testing watchlist stocks...")
    try:
        response = requests.get(f"{base_url}/api/watchlist-stocks")
        if response.status_code == 200:
            result = response.json()
            print("✅ Watchlist stocks successful")
            print(f"Number of stocks: {len(result)}")
            for stock in result[:3]:
                print(f"  - {stock['symbol']} ({stock['company']}): ${stock['price']} ({stock['changePercent']:+.2f}%)")
        else:
            print(f"❌ Watchlist stocks failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Watchlist stocks error: {e}")
    
    # Test 4: Individual stock data
    print("\n4. Testing individual stock data (VCB)...")
    try:
        response = requests.get(f"{base_url}/api/stock/VCB")
        if response.status_code == 200:
            result = response.json()
            print("✅ Individual stock data successful")
            print(f"Stock: {result['symbol']} - {result['company']}")
            print(f"Price: ${result['price']}")
            print(f"Change: ${result['change']} ({result['changePercent']:+.2f}%)")
        else:
            print(f"❌ Individual stock data failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Individual stock data error: {e}")
    
    # Test 5: Chart data
    print("\n5. Testing chart data (AAPL)...")
    try:
        response = requests.get(f"{base_url}/api/chart-data/AAPL")
        if response.status_code == 200:
            result = response.json()
            print("✅ Chart data successful")
            print(f"Data points: {len(result)}")
            if result:
                print(f"First point: {result[0]['time']} - ${result[0]['price']}")
                print(f"Last point: {result[-1]['time']} - ${result[-1]['price']}")
        else:
            print(f"❌ Chart data failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Chart data error: {e}")
    
    # Test 6: Financial news
    print("\n6. Testing financial news...")
    try:
        response = requests.get(f"{base_url}/api/news?limit=3")
        if response.status_code == 200:
            result = response.json()
            print("✅ Financial news successful")
            print(f"News items: {len(result)}")
            for news in result[:2]:
                print(f"  - {news['title'][:50]}... ({news['source']})")
        else:
            print(f"❌ Financial news failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Financial news error: {e}")
    
    # Test 7: Stock predictions
    print("\n7. Testing stock predictions (AAPL)...")
    try:
        response = requests.get(f"{base_url}/api/predictions/AAPL")
        if response.status_code == 200:
            result = response.json()
            print("✅ Stock predictions successful")
            print(f"Symbol: {result['symbol']}")
            print(f"Current: ${result['current_price']}")
            print(f"Predicted: ${result['predicted_price']} (30 days)")
            print(f"Confidence: {result['confidence']:.1%}")
        else:
            print(f"❌ Stock predictions failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stock predictions error: {e}")
    
    # Test 8: Portfolio data
    print("\n8. Testing portfolio data...")
    try:
        response = requests.get(f"{base_url}/api/portfolio")
        if response.status_code == 200:
            result = response.json()
            print("✅ Portfolio data successful")
            print(f"Total Value: ${result['total_value']:,.2f}")
            print(f"Total Return: {result['total_return_percent']:+.2f}%")
            print(f"Positions: {len(result['positions'])}")
        else:
            print(f"❌ Portfolio data failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Portfolio data error: {e}")
    
    print("\n🎉 Direct Finance API testing completed!")
    print("\n📋 Frontend Integration Instructions:")
    print("Replace your mock data imports with API calls:")
    print("- mockMarketData → GET /api/market-data/VN-INDEX")
    print("- mockWatchlistStocks → GET /api/watchlist-stocks")
    print("- mockNewsItems → GET /api/news")
    print("- Chart data → GET /api/chart-data/{symbol}")
    print("- Predictions → GET /api/predictions/{symbol}")
    print("- Portfolio → GET /api/portfolio")

if __name__ == "__main__":
    # Wait a moment for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    test_direct_finance_api()
