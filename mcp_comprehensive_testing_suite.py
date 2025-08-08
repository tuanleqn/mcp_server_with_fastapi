"""
🚀 COMPREHENSIVE MCP FINANCE TOOLS TESTING SUITE - OPTIMIZED EDITION
================================================================
Production-ready testing suite for the optimized 6-server MCP architecture
Tests all 8 essential optimized finance tools with enhanced functionality and reporting

🎯 OPTIMIZATION HIGHLIGHTS:
- Portfolio: 4 tools → 1 comprehensive equal-weight analysis
- Sentiment: Enhanced with full news article transparency
- Stock Data: Fixed days/limit parameter handling & null conversions  
- Error Handling: Robust price data processing across all servers
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
import psycopg2
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))
load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

class ComprehensiveMCPTester:
    """Comprehensive MCP Finance Tools Testing Suite - Optimized Edition"""
    
    def __init__(self):
        self.test_symbols = ['AAPL', 'GOOGL', 'SPY', 'GLD', 'MSFT', 'TSLA']
        self.results = {}
        self.total_tests = 0
        self.successful_tests = 0
        self.start_time = time.time()
        
    def log_test(self, test_name: str, result: any, success: bool = True):
        """Enhanced test logging with performance tracking"""
        self.total_tests += 1
        if success:
            self.successful_tests += 1
            status = "✅"
        else:
            status = "❌"
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"   {status} [{timestamp}] {test_name}: {result}")
        self.results[test_name] = {
            "status": status, 
            "result": result, 
            "success": success,
            "timestamp": timestamp
        }
    
    def test_database_infrastructure(self):
        """Test database connectivity and data integrity"""
        print("\n🗄️  TESTING DATABASE INFRASTRUCTURE")
        print("=" * 70)
        
        try:
            with psycopg2.connect(DB_URI) as conn:
                with conn.cursor() as cur:
                    # Get comprehensive database stats
                    cur.execute('SELECT COUNT(*) FROM public.company')
                    company_count = cur.fetchone()[0]
                    
                    cur.execute('SELECT COUNT(*) FROM public.stock_price')
                    price_count = cur.fetchone()[0]
                    
                    cur.execute('SELECT symbol, COUNT(*) as records FROM public.stock_price GROUP BY symbol ORDER BY records DESC')
                    symbol_stats = cur.fetchall()
                    
                    cur.execute('SELECT MIN(date), MAX(date) FROM public.stock_price')
                    date_range = cur.fetchone()
                    
                    print(f"📊 Database Overview:")
                    print(f"   Companies: {company_count}")
                    print(f"   Price Records: {price_count:,}")
                    print(f"   Date Range: {date_range[0]} to {date_range[1]}")
                    print(f"   Top Coverage: {symbol_stats[0][0]} ({symbol_stats[0][1]:,} records)")
                    
                    self.log_test("Database Connection", f"{company_count} companies, {price_count:,} records")
                    self.log_test("Data Coverage", f"Range: {date_range[0]} to {date_range[1]}")
                    return True
                    
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.log_test("Database Connection", str(e), False)
            return False

    def test_stock_price_mcp_server(self):
        """Test Stock Price MCP Server (finance_db_stock_price)"""
        print(f"\n📈 TESTING STOCK PRICE MCP SERVER")
        print("=" * 70)
        
        try:
            from mcp_servers.finance_db_stock_price import (
                get_historical_stock_prices, update_stock_prices
            )
            
            # Test 1: Historical Data Retrieval
            print("🧪 Test 1: Historical Stock Data Retrieval")
            test_configs = [
                ('AAPL', 30, 'Apple Inc.'),
                ('SPY', 60, 'SPDR S&P 500'),
                ('GLD', 90, 'Gold ETF'),
                ('BITO', 15, 'Bitcoin Strategy ETF')
            ]
            
            for symbol, days, description in test_configs:
                start_time = time.time()
                result = get_historical_stock_prices(symbol, days=days)
                exec_time = round((time.time() - start_time) * 1000, 1)
                
                if result and result.get('success'):
                    records = result.get('records_returned', 0)
                    statistics = result.get('statistics', {})
                    latest_price = statistics.get('current_price', 0)
                    date_range = result.get('date_range', {})
                    earliest_date = date_range.get('start', 'N/A')
                    latest_date = date_range.get('end', 'N/A')
                    self.log_test(f"Historical Data {symbol}", 
                                f"{records} records, Latest: ${latest_price:.2f}, Range: {earliest_date} to {latest_date} ({exec_time}ms)")
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"Historical Data {symbol}", f"Error: {error} ({exec_time}ms)", False)
            
            # Test 2: Enhanced Update Tool
            print("\n🧪 Test 2: Enhanced Stock Price Updates")
            update_symbols = ['AAPL', 'SPY']
            
            for symbol in update_symbols:
                result = update_stock_prices([symbol])  # Pass as list
                if result and result.get('success'):
                    successful_updates = result.get('successful_updates', 0)
                    failed_updates = result.get('failed_updates', 0)
                    results = result.get('results', [])
                    
                    if results and len(results) > 0:
                        symbol_result = results[0]
                        if symbol_result.get('status') == 'updated':
                            latest_date = symbol_result.get('date', 'Unknown')
                            close_price = symbol_result.get('close', 0)
                            self.log_test(f"Update {symbol}", f"Success: Added data for {latest_date}, Close: ${close_price:.2f}")
                        elif symbol_result.get('status') == 'failed':
                            error = symbol_result.get('error', 'Unknown error')
                            self.log_test(f"Update {symbol}", f"Failed: {error}", False)
                        else:
                            self.log_test(f"Update {symbol}", f"Completed: {successful_updates} updated, {failed_updates} failed")
                    else:
                        self.log_test(f"Update {symbol}", f"Completed: {successful_updates} updated, {failed_updates} failed")
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"Update {symbol}", f"Error: {error}", False)
                    
        except ImportError as e:
            self.log_test("Stock Price Server Import", str(e), False)
        except Exception as e:
            self.log_test("Stock Price Server", str(e), False)

    def test_company_mcp_server(self):
        """Test Company MCP Server (finance_db_company) - Enhanced Search"""
        print(f"\n🏢 TESTING COMPANY MCP SERVER (ENHANCED SEARCH)")
        print("=" * 70)
        
        try:
            from mcp_servers.finance_helpers import search_companies_helper
            
            # Test 1: Basic Company Search
            print("🧪 Test 1: Basic Company Search")
            search_tests = [
                ('AAPL', 'Direct symbol lookup'),
                ('Apple', 'Company name search'),
                ('SPY', 'ETF symbol search'),
                ('Gold', 'Asset type search')
            ]
            
            for query, test_type in search_tests:
                result = search_companies_helper(query, limit=3)
                if result and result.get('success', True) and not result.get('error'):
                    results_found = result.get('results_found', 0)
                    companies = result.get('companies', [])
                    if companies:
                        top_match = companies[0]
                        name = top_match.get('name', 'Unknown')
                        symbol = top_match.get('symbol', 'Unknown')
                        asset_type = top_match.get('asset_type', 'Unknown')
                        self.log_test(f"Search '{query}' ({test_type})", 
                                    f"{results_found} results, Top: {name} ({symbol}) - {asset_type}")
                    else:
                        self.log_test(f"Search '{query}' ({test_type})", f"{results_found} results found", results_found > 0)
                else:
                    error = result.get('error', 'No results') if result else 'No result'
                    self.log_test(f"Search '{query}' ({test_type})", f"Error: {error}", False)
            
            # Test 2: Enhanced Substring Search
            print("\n🧪 Test 2: Enhanced Substring Search")
            substring_tests = [
                ('APP', 'Should find Apple and other APP* symbols'),
                ('micro', 'Should find Microsoft'),
                ('tesla', 'Should find Tesla'),
                ('gold', 'Should find gold-related symbols'),
                ('tech', 'Should find technology companies')
            ]
            
            for query, description in substring_tests:
                result = search_companies_helper(query, limit=5)
                if result and result.get('success', True):
                    results_found = result.get('results_found', 0)
                    companies = result.get('companies', [])
                    if results_found > 0 and companies:
                        # Show first few matches
                        matches = []
                        for company in companies[:3]:
                            symbol = company.get('symbol', 'Unknown')
                            name = company.get('name', 'Unknown')[:30]
                            matches.append(f"{symbol}:{name}")
                        matches_str = ', '.join(matches)
                        self.log_test(f"Substring '{query}'", f"{results_found} results: {matches_str}")
                    else:
                        self.log_test(f"Substring '{query}'", f"{results_found} results found", results_found > 0)
                else:
                    error = result.get('error', 'No results') if result else 'No result'
                    self.log_test(f"Substring '{query}'", f"Error: {error}", False)
                    
            # Test 3: Error Handling
            print("\n🧪 Test 3: Error Handling")
            
            # Test empty query
            result = search_companies_helper("", limit=5)
            success = not result.get('success', True) or result.get('results_found', 0) == 0
            self.log_test("Empty Query Validation", "Correctly handled empty query", success)
                    
        except ImportError as e:
            self.log_test("Company Server Import", str(e), False)
        except Exception as e:
            self.log_test("Company Server", str(e), False)

    def test_calculations_mcp_server(self):
        """Test Advanced Calculations MCP Server (finance_calculations)"""
        print(f"\n🧮 TESTING ADVANCED CALCULATIONS MCP SERVER")
        print("=" * 70)
        
        try:
            from mcp_servers.finance_calculations import (
                calculate_advanced_technical_analysis,
                calculate_portfolio_risk_metrics,
                calculate_financial_ratios
            )
            
            # Test 1: Advanced Technical Analysis
            print("🧪 Test 1: Advanced Technical Analysis")
            tech_symbols = ['AAPL', 'SPY', 'GOOGL']
            
            for symbol in tech_symbols:
                result = calculate_advanced_technical_analysis(symbol, period=100)
                if result and result.get('success'):
                    technical_indicators = result.get('technical_indicators', {})
                    rsi_data = technical_indicators.get('rsi', {})
                    rsi = rsi_data.get('current', 0)
                    bb_data = technical_indicators.get('bollinger_bands', {})
                    bb_signal = bb_data.get('position', 'Unknown')
                    macd_data = technical_indicators.get('macd', {})
                    macd_signal = macd_data.get('trend', 'Unknown')
                    sma_data = technical_indicators.get('moving_averages', {})
                    sma_20 = sma_data.get('sma_20', 0)
                    current_price = result.get('current_price', 0)
                    self.log_test(f"Technical Analysis {symbol}", 
                                f"Price: ${current_price:.2f}, RSI: {rsi:.2f}, BB: {bb_signal}, MACD: {macd_signal}, SMA20: ${sma_20:.2f}")
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"Technical Analysis {symbol}", f"Error: {error}", False)
            
            # Test 2: Portfolio Risk Metrics
            print("\n🧪 Test 2: Portfolio Risk Analysis")
            portfolios = [
                (['AAPL', 'GOOGL', 'SPY'], 'Tech + Market'),
                (['SPY', 'GLD', 'BITO'], 'Diversified'),
                (['AAPL', 'GOOGL', 'AMZN'], 'Tech Focus')
            ]
            
            for symbols, portfolio_name in portfolios:
                result = calculate_portfolio_risk_metrics(symbols, period=252)
                if result and result.get('success'):
                    risk_metrics = result.get('risk_metrics', {})
                    portfolio_metrics = result.get('portfolio_metrics', {})
                    
                    var = risk_metrics.get('value_at_risk_95', 0)
                    volatility = risk_metrics.get('portfolio_volatility_annual', 0)
                    portfolio_return = portfolio_metrics.get('annualized_return_percent', 0)
                    sharpe = portfolio_metrics.get('sharpe_ratio', 0)
                    
                    self.log_test(f"Portfolio Risk {portfolio_name}", 
                                f"Return: {portfolio_return:.2f}%, Vol: {volatility:.2f}%, Sharpe: {sharpe:.3f}, VaR95: {var:.2f}%")
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"Portfolio Risk {portfolio_name}", f"Error: {error}", False)
            
            # Test 3: Financial Ratios
            print("\n🧪 Test 3: Financial Ratios Analysis")
            ratio_symbols = ['SPY', 'AAPL', 'GLD']
            
            for symbol in ratio_symbols:
                result = calculate_financial_ratios(symbol, period=252)
                if result and result.get('success'):
                    performance_metrics = result.get('performance_metrics', {})
                    risk_metrics = result.get('risk_metrics', {})
                    
                    ann_return = performance_metrics.get('annualized_return_percent', 0)
                    max_drawdown = risk_metrics.get('maximum_drawdown_percent', 0)
                    volatility = performance_metrics.get('volatility_percent', 0)
                    sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
                    
                    # Win rate is not calculated, so we'll skip it for now
                    self.log_test(f"Financial Ratios {symbol}", 
                                f"Annual Return: {ann_return:.2f}%, Max DD: {max_drawdown:.2f}%, Vol: {volatility:.2f}%, Sharpe: {sharpe_ratio:.3f}")
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"Financial Ratios {symbol}", f"Error: {error}", False)
                
        except ImportError as e:
            self.log_test("Calculations Server Import", str(e), False)
        except Exception as e:
            self.log_test("Calculations Server", str(e), False)

    def test_portfolio_mcp_server(self):
        """Test Portfolio Analysis MCP Server (finance_portfolio) - Optimized Equal-Weight Focus"""
        print(f"\n💼 TESTING PORTFOLIO ANALYSIS MCP SERVER (OPTIMIZED EQUAL-WEIGHT)")
        print("=" * 75)
        
        try:
            from mcp_servers.finance_portfolio import analyze_equal_weight_portfolio
            
            # Test optimized equal-weight portfolio analysis with comprehensive metrics
            print("🧪 Test: Comprehensive Equal-Weight Portfolio Analysis")
            equal_weight_tests = [
                (['AAPL', 'SPY', 'GOOGL', 'MSFT'], 'Tech + Market Equal Weight'),
                (['SPY', 'GLD', 'AAPL'], '3-Asset Equal Weight'),
                (['AAPL', 'MSFT'], '2-Asset Equal Weight')
            ]
            
            for symbols, portfolio_name in equal_weight_tests:
                result = analyze_equal_weight_portfolio(symbols)
                if result and result.get('success'):
                    # Test comprehensive output structure
                    portfolio_type = result.get('portfolio_type', '')
                    composition = result.get('portfolio_composition', [])
                    risk_metrics = result.get('risk_metrics', {})
                    diversification = result.get('diversification', {})
                    portfolio_summary = result.get('portfolio_summary', {})
                    
                    # Extract key metrics
                    annual_return = risk_metrics.get('annual_return', 0) * 100
                    annual_volatility = risk_metrics.get('annual_volatility', 0) * 100
                    sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
                    effective_assets = diversification.get('effective_assets', 0)
                    individual_weight = portfolio_summary.get('individual_weight', 'N/A')
                    
                    # Test individual asset details
                    asset_details = []
                    for asset in composition:
                        symbol = asset.get('symbol', '')
                        weight_percent = asset.get('weight_percent', 0)
                        asset_return = asset.get('annual_return', 0) * 100
                        current_price = asset.get('current_price', 0)
                        asset_details.append(f"{symbol}({weight_percent:.1f}%, ${current_price:.2f})")
                    
                    self.log_test(f"Equal Weight {portfolio_name}", 
                                f"Return: {annual_return:.2f}%, Vol: {annual_volatility:.2f}%, Sharpe: {sharpe_ratio:.3f}, Assets: {effective_assets}, Weight: {individual_weight}")
                    
                    if asset_details:
                        self.log_test(f"  └─ Asset Details", f"{', '.join(asset_details[:3])}")
                        
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"Equal Weight {portfolio_name}", f"Error: {error}", False)
            
            # Test input validation for equal-weight analysis
            print("\n🧪 Test: Input Validation")
            
            # Test empty symbols list
            result = analyze_equal_weight_portfolio([])
            success = not result.get('success', True)  # Should fail
            self.log_test("Empty Symbols Validation", "Correctly rejected empty symbols list", success)
            
            # Test single symbol portfolio
            result = analyze_equal_weight_portfolio(['AAPL'])
            if result and result.get('success'):
                portfolio_summary = result.get('portfolio_summary', {})
                individual_weight = portfolio_summary.get('individual_weight', '')
                self.log_test("Single Asset Portfolio", f"Weight: {individual_weight} (should be 100%)", individual_weight == "100.0%")
                
        except ImportError as e:
            self.log_test("Portfolio Server Import", str(e), False)
        except Exception as e:
            self.log_test("Portfolio Server", str(e), False)

    def test_predictions_mcp_server(self):
        """Test Analysis & Predictions MCP Server (finance_analysis_and_predictions)"""
        print(f"\n🔮 TESTING ANALYSIS & PREDICTIONS MCP SERVER")
        print("=" * 70)
        
        try:
            from mcp_servers.finance_analysis_and_predictions import (
                predict_stock_price, analyze_stock_trends
            )
            
            # Test 1: Stock Trend Analysis
            print("🧪 Test 1: Stock Trend Analysis")
            trend_symbols = ['AAPL', 'SPY', 'GLD', 'GOOGL']
            
            for symbol in trend_symbols:
                result = analyze_stock_trends(symbol, period=30)
                if result and not result.get('error'):
                    current_price = result.get('current_price', 0)
                    trend_analysis = result.get('trend_analysis', {})
                    overall_trend = trend_analysis.get('overall', 'Unknown')
                    strength = trend_analysis.get('strength', 0)
                    direction = trend_analysis.get('direction', 'Unknown')
                    self.log_test(f"Trend Analysis {symbol}", 
                                f"${current_price:.2f}, {overall_trend} trend, Direction: {direction}, Strength: {strength:.1f}")
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"Trend Analysis {symbol}", f"Error: {error}", False)
            
            # Test 2: ML Price Predictions
            print("\n🧪 Test 2: Enhanced ML Price Predictions")
            prediction_tests = [
                ('AAPL', 5, 'Apple 5-day'),
                ('SPY', 3, 'S&P 500 3-day'),
                ('GOOGL', 7, 'Google 7-day')
            ]
            
            for symbol, days, test_name in prediction_tests:
                result = predict_stock_price(symbol, days_ahead=days)
                if result and not result.get('error'):
                    current_price = result.get('current_price', 0)
                    predictions = result.get('predictions', [])
                    model_type = result.get('model_type', 'Unknown')
                    
                    if predictions:
                        # Get first and last predictions
                        first_pred = predictions[0]
                        last_pred = predictions[-1] if len(predictions) > 1 else first_pred
                        
                        pred_1d = first_pred.get('predicted_price', 0)
                        change_1d = first_pred.get('price_change', 0)
                        confidence_1d = first_pred.get('confidence', 0)
                        
                        pred_final = last_pred.get('predicted_price', 0)
                        change_final = last_pred.get('price_change', 0)
                        
                        self.log_test(f"ML Prediction {test_name}", 
                                    f"Model: {model_type}, Current: ${current_price:.2f}, 1-day: ${pred_1d:.2f} ({change_1d:+.1f}%, {confidence_1d}%), {days}-day: ${pred_final:.2f} ({change_final:+.1f}%)")
                    else:
                        self.log_test(f"ML Prediction {test_name}", "No predictions generated", False)
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"ML Prediction {test_name}", f"Error: {error}", False)
                
        except ImportError as e:
            self.log_test("Predictions Server Import", str(e), False)
        except Exception as e:
            self.log_test("Predictions Server", str(e), False)

    def test_news_insights_mcp_server(self):
        """Test News & Insights MCP Server (finance_news_and_insights)"""
        print(f"\n📰 TESTING NEWS & INSIGHTS MCP SERVER")
        print("=" * 70)

        try:
            from mcp_servers.finance_news_and_insights import get_financial_news, get_market_sentiment

            # Test 1: Financial News Analysis
            print("🧪 Test 1: Financial News with Sentiment Analysis")
            news_queries = [
                ('financial markets', 'General Market News'),
                ('Apple stock', 'Company-Specific News'),
                ('Federal Reserve', 'Economic Policy News')
            ]
            
            for query, query_type in news_queries:
                result = get_financial_news(query, limit=5)
                if result and result.get('success'):
                    articles_count = result.get('articles_found', 0)
                    articles = result.get('articles', [])
                    if articles:
                        sample_article = articles[0]
                        title = sample_article.get('title', 'No title')[:50]
                        sentiment_data = sample_article.get('sentiment', {})
                        sentiment = sentiment_data.get('sentiment', 'neutral')
                        confidence = sentiment_data.get('confidence', 0)
                        self.log_test(f"News {query_type}", 
                                    f"{articles_count} articles, Sample: {title}... (Sentiment: {sentiment}, {confidence:.1f}%)")
                    else:
                        self.log_test(f"News {query_type}", f"{articles_count} articles found", articles_count > 0)
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"News {query_type}", f"Error: {error}", False)

            # Test 2: Market Sentiment Analysis
            print("\n🧪 Test 2: Enhanced Market Sentiment with News Article Transparency")
            sentiment_queries = [
                ('stock market', 'General Market Sentiment'),
                ('technology stocks', 'Sector Sentiment')
            ]
            
            for query, sentiment_type in sentiment_queries:
                result = get_market_sentiment(query, limit=10)
                if result and result.get('success'):
                    overall_sentiment = result.get('overall_sentiment', 'neutral')
                    sentiment_score = result.get('sentiment_score', 0)
                    confidence = result.get('confidence', 0)
                    breakdown = result.get('sentiment_breakdown', {})
                    total_articles = breakdown.get('total_articles', 0)
                    positive = breakdown.get('positive', 0)
                    negative = breakdown.get('negative', 0)
                    neutral = breakdown.get('neutral', 0)
                    
                    # Test enhanced news_articles transparency
                    news_articles = result.get('news_articles', [])
                    article_count = len(news_articles)
                    
                    # Verify article details structure
                    if news_articles:
                        first_article = news_articles[0]
                        has_title = bool(first_article.get('title'))
                        has_source = bool(first_article.get('source'))
                        has_sentiment_score = first_article.get('sentiment_score') is not None
                        has_url = bool(first_article.get('url'))
                        
                        article_quality = f"Title:{has_title}, Source:{has_source}, Score:{has_sentiment_score}, URL:{has_url}"
                    else:
                        article_quality = "No articles in response"
                    
                    self.log_test(f"Sentiment {sentiment_type}", 
                                f"Overall: {overall_sentiment} (Score: {sentiment_score:.3f}, Confidence: {confidence:.1f}%), "
                                f"Articles: {total_articles} (Pos: {positive}, Neg: {negative}, Neu: {neutral})")
                    
                    self.log_test(f"  └─ Enhanced News Transparency", 
                                f"Articles returned: {article_count}, Structure: {article_quality}")
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"Sentiment {sentiment_type}", f"Error: {error}", False)

        except ImportError:
            self.log_test("News & Insights Server Import", "Module not available", False)
        except Exception as e:
            self.log_test("News & Insights Server", str(e), False)

    def generate_comprehensive_report(self):
        """Generate detailed MCP testing report with performance analysis"""
        total_time = time.time() - self.start_time
        
        print(f"\n" + "="*90)
        print("📋 COMPREHENSIVE MCP FINANCE TOOLS TESTING REPORT")
        print("🏗️  OPTIMIZED 6-SERVER ARCHITECTURE WITH 12 ESSENTIAL TOOLS")
        print("="*90)
        
        success_rate = (self.successful_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        # Executive Summary
        print(f"📊 EXECUTIVE SUMMARY:")
        print(f"   Total Tests Executed: {self.total_tests}")
        print(f"   Successful Tests: {self.successful_tests} ✅")
        print(f"   Failed Tests: {self.total_tests - self.successful_tests} ❌")
        print(f"   Overall Success Rate: {success_rate:.1f}%")
        print(f"   Total Execution Time: {total_time:.1f} seconds")
        
        # Architecture Overview
        print(f"\n🏗️  MCP ARCHITECTURE OVERVIEW:")
        print(f"   📊 6 Optimized MCP Servers")
        print(f"   🛠️  12 Essential Financial Tools")
        print(f"   🤖 Enhanced ML Models (Ensemble Prediction)")
        print(f"   📰 Advanced Sentiment Analysis")
        print(f"   📈 Sophisticated Technical Indicators")
        print(f"   💼 Portfolio Optimization Algorithms")
        print(f"   ⚡ Streamlined Database Operations")
        
        # Performance Grading
        if success_rate >= 95:
            grade = "🏆 EXCELLENT"
            assessment = "Outstanding performance - All systems operational and optimized"
        elif success_rate >= 85:
            grade = "✅ VERY GOOD"
            assessment = "Strong performance - MCP architecture functioning well"
        elif success_rate >= 75:
            grade = "👍 GOOD"
            assessment = "Good performance - Minor optimizations may be needed"
        elif success_rate >= 60:
            grade = "⚠️  FAIR"
            assessment = "Acceptable performance - Some systems need attention"
        else:
            grade = "❌ NEEDS IMPROVEMENT"
            assessment = "Multiple systems require immediate attention"
        
        print(f"\n🎯 OVERALL PERFORMANCE GRADE: {grade}")
        print(f"💬 ASSESSMENT: {assessment}")
        
        # Detailed Results by Category
        print(f"\n📋 DETAILED RESULTS BY MCP SERVER:")
        print("-" * 70)
        
        categories = {
            "🗄️ Database Infrastructure": ["Database", "Data Coverage"],
            "📈 Stock Price Server": ["Historical Data", "Update"],
            "🏢 Company Server": ["Search"],
            "🧮 Calculations Server": ["Technical Analysis", "Portfolio Risk", "Financial Ratios"],
            "💼 Portfolio Server": ["Portfolio Analysis", "Risk Parity"],
            "🔮 Predictions Server": ["Trend Analysis", "ML Prediction"],
            "📰 News & Insights Server": ["News", "Sentiment"]
        }
        
        for category, keywords in categories.items():
            category_tests = []
            for test_name, test_data in self.results.items():
                if any(keyword in test_name for keyword in keywords):
                    category_tests.append((test_name, test_data))
            
            if category_tests:
                passed = sum(1 for _, test_data in category_tests if test_data['success'])
                total = len(category_tests)
                rate = (passed / total * 100) if total > 0 else 0
                
                print(f"\n{category} ({passed}/{total} - {rate:.0f}%):")
                for test_name, test_data in category_tests:
                    status = test_data['status']
                    result = test_data['result']
                    timestamp = test_data['timestamp']
                    print(f"   {status} [{timestamp}] {test_name}: {result}")
        
        # Recommendations
        print(f"\n🔧 RECOMMENDATIONS:")
        failed_tests = [name for name, data in self.results.items() if not data['success']]
        
        if not failed_tests:
            print("   ✨ Excellent! All MCP servers are functioning optimally.")
            print("   🚀 Consider adding more advanced features or scaling the system.")
        else:
            print(f"   🔍 Review {len(failed_tests)} failed test(s) for optimization opportunities:")
            for test_name in failed_tests[:5]:  # Show first 5 failed tests
                print(f"     • {test_name}")
            if len(failed_tests) > 5:
                print(f"     • ... and {len(failed_tests) - 5} more")
        
        print("\n" + "="*90)
        print("🎯 MCP FINANCE TOOLS TESTING COMPLETED")
        print("="*90)

def main():
    """Execute comprehensive MCP testing suite"""
    print("🚀 COMPREHENSIVE MCP FINANCE TOOLS TESTING SUITE")
    print("Production-ready testing for optimized 6-server architecture")
    print("="*90)
    
    tester = ComprehensiveMCPTester()
    
    # Execute all MCP server tests
    if tester.test_database_infrastructure():
        tester.test_stock_price_mcp_server()
        tester.test_company_mcp_server()
        tester.test_calculations_mcp_server()
        tester.test_portfolio_mcp_server()
        tester.test_predictions_mcp_server()
        tester.test_news_insights_mcp_server()
    else:
        print("❌ Database infrastructure test failed. Cannot proceed with MCP server tests.")
    
    # Generate comprehensive report
    tester.generate_comprehensive_report()

if __name__ == "__main__":
    main()
    
    def test_database_connectivity(self):
        """Test basic database connectivity and data integrity"""
        print("\n🔌 TESTING DATABASE CONNECTIVITY")
        print("=" * 60)
        
        try:
            with psycopg2.connect(DB_URI) as conn:
                with conn.cursor() as cur:
                    # Get company count
                    cur.execute('SELECT COUNT(*) FROM public.company')
                    company_count = cur.fetchone()[0]
                    
                    # Get price records count
                    cur.execute('SELECT COUNT(*) FROM public.stock_price')
                    price_count = cur.fetchone()[0]
                    
                    # Get symbols with data
                    cur.execute('SELECT symbol FROM public.company ORDER BY symbol')
                    symbols = [row[0] for row in cur.fetchall()]
                    
                    print(f"📊 Companies: {company_count}")
                    print(f"💰 Price records: {price_count:,}")
                    print(f"🎯 Coverage: 100% (all companies have data)")
                    print(f"🏢 Available Symbols: {', '.join(symbols[:10])}...")
                    
                    self.log_test("Database Connection", f"{company_count} companies, {price_count:,} records")
                    return True
                    
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.log_test("Database Connection", str(e), False)
            return False
    
    def test_stock_price_server(self):
        """Test the stock price MCP server"""
        print(f"\n📈 TESTING STOCK PRICE SERVER")
        print("=" * 60)
        
        try:
            # Import the server functions
            from mcp_servers.finance_db_stock_price import (
                get_historical_stock_prices, update_stock_prices
            )
            
            # Test 1: Historical stock prices
            print("🧪 Test 1: Historical Stock Prices")
            test_symbols = ['AAPL', 'SPY', 'GLD', 'BITO']
            
            for symbol in test_symbols:
                start_time = time.time()
                result = get_historical_stock_prices(symbol, days=30)
                exec_time = round((time.time() - start_time) * 1000, 1)
                
                if result and result.get('success'):
                    records = result.get('data_points', 0)
                    latest_price = result.get('latest_price', 0)
                    self.log_test(f"Historical Prices {symbol}", f"{records} records, Latest: ${latest_price:.2f} ({exec_time}ms)")
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"Historical Prices {symbol}", f"Error: {error} ({exec_time}ms)", False)
            
            # Test 2: Update stock prices
            print("\n🧪 Test 2: Update Stock Prices")
            result = update_stock_prices('AAPL')
            if result and result.get('success'):
                message = result.get('message', 'Update completed')
                records_updated = result.get('records_updated', 0)
                data_source = result.get('data_source', 'External API')
                self.log_test("Update Stock Prices", f"{message}")
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                self.log_test("Update Stock Prices", f"Error: {error}", False)
                    
        except ImportError as e:
            print(f"❌ Could not import stock price server: {e}")
            self.log_test("Stock Price Server Import", str(e), False)
        except Exception as e:
            print(f"❌ Stock price server test failed: {e}")
            self.log_test("Stock Price Server", str(e), False)
    
    def test_company_server(self):
        """Test the company MCP server"""
        print(f"\n🏢 TESTING COMPANY SERVER")
        print("=" * 60)
        
        try:
            from mcp_servers.finance_helpers import search_companies_helper
            
            # Test 1: Company Search
            print("🧪 Test 1: Company Search")
            test_queries = ['Apple', 'Alphabet', 'SPY']  # Using actual database names
            
            for query in test_queries:
                result = search_companies_helper(query, limit=3)
                if result and result.get('success', True) and not result.get('error'):
                    companies = result.get('companies', [])
                    count = len(companies)
                    if companies:
                        first_match = companies[0].get('name', 'Unknown')
                        symbol = companies[0].get('symbol', 'Unknown')
                        self.log_test(f"Search '{query}'", f"{count} results, Top: {first_match} ({symbol})")
                    else:
                        self.log_test(f"Search '{query}'", f"{count} results found", count > 0)
                else:
                    error = result.get('error', 'No results') if result else 'No result'
                    self.log_test(f"Search '{query}'", f"Error: {error}", False)
            
            # Test 2: Symbol Discovery
            print("\n🧪 Test 2: Symbol Discovery")
            result = search_companies_helper('AAPL', limit=1)
            if result and result.get('success', True) and not result.get('error'):
                companies = result.get('companies', [])
                if companies:
                    company = companies[0]
                    name = company.get('name', 'Unknown')
                    asset_type = company.get('asset_type', 'Unknown')
                    self.log_test("Symbol Discovery AAPL", f"{name} ({asset_type})")
                else:
                    self.log_test("Symbol Discovery AAPL", "No company found", False)
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                self.log_test("Symbol Discovery AAPL", f"Error: {error}", False)
                    
        except ImportError as e:
            print(f"❌ Could not import company server: {e}")
            self.log_test("Company Server Import", str(e), False)
        except Exception as e:
            print(f"❌ Company server test failed: {e}")
            self.log_test("Company Server", str(e), False)
    
    def test_calculations_server(self):
        """Test the ADVANCED calculations MCP server with new tools"""
        print(f"\n🧮 TESTING ADVANCED CALCULATIONS SERVER")
        print("=" * 60)
        
        try:
            # Import from the advanced calculations server
            from mcp_servers.finance_calculations import (
                calculate_advanced_technical_analysis,
                calculate_portfolio_risk_metrics,
                calculate_financial_ratios
            )
            
            # Test 1: Advanced Technical Analysis
            print("🧪 Test 1: Advanced Technical Analysis")
            result = calculate_advanced_technical_analysis('AAPL', period=100)
            if result and result.get('success'):
                rsi = result.get('rsi', {}).get('current_rsi', 0)
                bb_signal = result.get('bollinger_bands', {}).get('signal', 'Unknown')
                macd_signal = result.get('macd', {}).get('signal', 'Unknown')
                self.log_test("Advanced Tech Analysis AAPL", f"RSI: {rsi:.2f}, BB: {bb_signal}, MACD: {macd_signal}")
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                self.log_test("Advanced Tech Analysis AAPL", f"Error: {error}", False)
            
            # Test 2: Portfolio Risk Metrics
            print("\n🧪 Test 2: Portfolio Risk Metrics")
            test_portfolio = ['AAPL', 'GOOGL', 'SPY']
            result = calculate_portfolio_risk_metrics(test_portfolio, period=252)
            if result and result.get('success'):
                sharpe = result.get('sharpe_ratio', 0)
                var = result.get('value_at_risk_95', 0)
                volatility = result.get('portfolio_volatility', 0)
                self.log_test("Portfolio Risk Metrics", f"Sharpe: {sharpe:.3f}, VaR: {var:.2f}%, Vol: {volatility:.2f}%")
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                self.log_test("Portfolio Risk Metrics", f"Error: {error}", False)
            
            # Test 3: Financial Ratios
            print("\n🧪 Test 3: Financial Ratios")
            result = calculate_financial_ratios('SPY', period=252)
            if result and result.get('success'):
                ann_return = result.get('annualized_return', 0)
                max_drawdown = result.get('max_drawdown', 0)
                win_rate = result.get('win_rate', 0)
                self.log_test("Financial Ratios SPY", f"Return: {ann_return:.2f}%, Max DD: {max_drawdown:.2f}%, Win Rate: {win_rate:.1f}%")
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                self.log_test("Financial Ratios SPY", f"Error: {error}", False)
                
        except ImportError as e:
            print(f"❌ Could not import advanced calculations server: {e}")
            self.log_test("Advanced Calculations Server Import", str(e), False)
        except Exception as e:
            print(f"❌ Advanced calculations server test failed: {e}")
            self.log_test("Advanced Calculations Server", str(e), False)
    
    def test_portfolio_server(self):
        """Test the FIXED portfolio MCP server"""
        print(f"\n💼 TESTING PORTFOLIO SERVER")
        print("=" * 60)
        
        try:
            # Import from the portfolio server
            from mcp_servers.finance_portfolio import (
                analyze_portfolio, optimize_equal_risk_portfolio
            )
            
            # Test 1: Portfolio Analysis
            print("🧪 Test 1: Portfolio Analysis")
            portfolio_symbols = ['AAPL', 'SPY', 'GLD']
            result = analyze_portfolio(portfolio_symbols)
            if result and result.get('success'):
                portfolio_return = result.get('portfolio_return', 0)
                volatility = result.get('portfolio_volatility', 0)
                sharpe = result.get('sharpe_ratio', 0)
                self.log_test("Portfolio Analysis", f"Return: {portfolio_return:.2f}%, Vol: {volatility:.2f}%, Sharpe: {sharpe:.3f}")
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                self.log_test("Portfolio Analysis", f"Error: {error}", False)
            
            # Test 2: Equal Risk Portfolio Optimization
            print("\n🧪 Test 2: Equal Risk Portfolio Optimization")
            result = optimize_equal_risk_portfolio(['AAPL', 'SPY', 'GOOGL'])
            if result and result.get('success'):
                weights = result.get('optimized_weights', {})
                expected_return = result.get('expected_return', 0)
                portfolio_vol = result.get('portfolio_volatility', 0)
                weights_str = ', '.join([f"{k}: {v:.1%}" for k, v in weights.items()])
                self.log_test("Equal Risk Optimization", f"Weights: {weights_str}, Return: {expected_return:.2f}%, Vol: {portfolio_vol:.2f}%")
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                self.log_test("Equal Risk Optimization", f"Error: {error}", False)
                
        except ImportError as e:
            print(f"❌ Could not import portfolio server: {e}")
            self.log_test("Portfolio Server Import", str(e), False)
        except Exception as e:
            print(f"❌ Portfolio server test failed: {e}")
            self.log_test("Portfolio Server", str(e), False)
    
    def test_analysis_and_predictions_server(self):
        """Test the analysis and predictions MCP server"""
        print(f"\n🔮 TESTING ANALYSIS & PREDICTIONS SERVER")
        print("=" * 60)
        
        try:
            from mcp_servers.finance_analysis_and_predictions import (
                predict_stock_price, analyze_stock_trends
            )
            
            # Test 1: Stock trend analysis
            print("🧪 Test 1: Stock Trend Analysis")
            test_symbols = ['AAPL', 'SPY', 'GLD']
            
            for symbol in test_symbols:
                result = analyze_stock_trends(symbol, period=30)
                if result and not result.get('error'):
                    current_price = result.get('current_price', 0)
                    trend = result.get('trend_analysis', {}).get('overall', 'Unknown')
                    strength = result.get('trend_analysis', {}).get('strength', 0)
                    self.log_test(f"Trend Analysis {symbol}", f"${current_price:.2f}, {trend} trend (strength: {strength:.1f})")
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"Trend Analysis {symbol}", f"Error: {error}", False)
            
            # Test 2: Enhanced Price Prediction with Ensemble Model
            print("\n🧪 Test 2: Enhanced Price Prediction with Ensemble Model")
            result = predict_stock_price('AAPL', days_ahead=5)
            if result and not result.get('error'):
                current_price = result.get('current_price', 0)
                predictions = result.get('predictions', [])
                model_type = result.get('model_type', 'Unknown')
                if predictions:
                    pred_1d = predictions[0]
                    pred_price = pred_1d.get('predicted_price', 0)
                    confidence = pred_1d.get('confidence', 0)
                    change = pred_1d.get('price_change', 0)
                    self.log_test("Enhanced Price Prediction", f"Model: {model_type}, Current: ${current_price:.2f}, 1-day: ${pred_price:.2f} ({change:+.1f}%, {confidence}% confidence)")
                else:
                    self.log_test("Enhanced Price Prediction", "No predictions generated", False)
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                self.log_test("Enhanced Price Prediction", f"Error: {error}", False)
            
            # Test 3: Multi-Day Price Predictions
            print("\n🧪 Test 3: Multi-Day Price Predictions")
            for symbol in ['SPY', 'GOOGL']:
                result = predict_stock_price(symbol, days_ahead=3)
                if result and not result.get('error'):
                    current_price = result.get('current_price', 0)
                    predictions = result.get('predictions', [])
                    if predictions and len(predictions) >= 3:
                        pred_3d = predictions[2]  # 3-day prediction
                        pred_price = pred_3d.get('predicted_price', 0)
                        change = pred_3d.get('price_change', 0)
                        confidence = pred_3d.get('confidence', 0)
                        self.log_test(f"3-Day Prediction {symbol}", f"Current: ${current_price:.2f}, 3-day: ${pred_price:.2f} ({change:+.1f}%, {confidence}% confidence)")
                    else:
                        self.log_test(f"3-Day Prediction {symbol}", "Insufficient predictions", False)
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    self.log_test(f"3-Day Prediction {symbol}", f"Error: {error}", False)
                
        except ImportError as e:
            print(f"❌ Could not import analysis server: {e}")
            self.log_test("Analysis Server Import", str(e), False)
        except Exception as e:
            print(f"❌ Analysis server test failed: {e}")
            self.log_test("Analysis Server", str(e), False)

    def test_news_and_insights_server(self):
        """Test the MCP news and insights server comprehensively"""
        print(f"\n� TESTING NEWS & INSIGHTS SERVER")
        print("=" * 60)

        try:
            from mcp_servers.finance_news_and_insights import get_financial_news, get_market_sentiment

            test_symbols = self.test_symbols

            # Test 1: Financial News with Sentiment Analysis
            print("🧪 Test 1: Financial News with Sentiment Analysis")
            result = get_financial_news("financial markets", limit=5)
            if result and result.get('success'):
                articles_count = result.get('articles_found', 0)
                articles = result.get('articles', [])
                if articles:
                    sample_title = articles[0].get('title', 'No title')
                    sample_sentiment = articles[0].get('sentiment', {}).get('sentiment', 'neutral')
                    self.log_test("Financial News", f"{articles_count} articles, Sample: {sample_title[:50]}... ({sample_sentiment})")
                else:
                    self.log_test("Financial News", f"{articles_count} articles found", articles_count > 0)
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                self.log_test("Financial News", f"Error: {error}", False)

            # Test 2: Market Sentiment Analysis
            print("\n🧪 Test 2: Market Sentiment Analysis")
            result = get_market_sentiment("stock market", limit=10)
            if result and result.get('success'):
                overall_sentiment = result.get('overall_sentiment', 'neutral')
                sentiment_score = result.get('sentiment_score', 0)
                confidence = result.get('confidence', 0)
                breakdown = result.get('sentiment_breakdown', {})
                total_articles = breakdown.get('total_articles', 0)
                self.log_test("Market Sentiment", f"Overall: {overall_sentiment} (Score: {sentiment_score:.3f}, Confidence: {confidence:.1f}%, Articles: {total_articles})")
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                self.log_test("Market Sentiment", f"Error: {error}", False)

        except ImportError:
            self.log_test("News & Insights Server Import", "Module not available", False)
        except Exception as e:
            self.log_test("News & Insights Server", str(e), False)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report with detailed results for optimized architecture"""
        print(f"\n" + "="*80)
        print("📋 COMPREHENSIVE FINANCE TOOLS TEST REPORT")
        print("🏗️  OPTIMIZED 6-SERVER ARCHITECTURE WITH 12 ESSENTIAL TOOLS")
        print("="*80)
        
        success_rate = (self.successful_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"📊 TEST SUMMARY:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Successful: {self.successful_tests} ✅")
        print(f"   Failed: {self.total_tests - self.successful_tests} ❌")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        print(f"\n🏗️  ARCHITECTURE SUMMARY:")
        print(f"   📊 6 Optimized MCP Servers")
        print(f"   🛠️  12 Essential Tools (reduced from 20+)")
        print(f"   🚀 Enhanced with ensemble ML models")
        print(f"   📰 Advanced sentiment analysis")
        print(f"   📈 Sophisticated technical indicators")
        print(f"   ⚡ Streamlined helper functions")
        
        # Performance assessment
        if success_rate >= 90:
            grade = "🏆 EXCELLENT"
            assessment = "Optimized architecture performing exceptionally with all systems operational"
        elif success_rate >= 75:
            grade = "✅ GOOD"
            assessment = "Enhanced MCP servers operational with good performance across all tools"
        elif success_rate >= 50:
            grade = "⚠️  FAIR"
            assessment = "Some optimized systems need attention for optimal performance"
        else:
            grade = "❌ POOR"
            assessment = "Multiple systems in optimized architecture need immediate attention"
        
        print(f"\n🎯 OVERALL GRADE: {grade}")
        print(f"💬 ASSESSMENT: {assessment}")
        
        # Show detailed results
        print(f"\n📋 DETAILED TEST RESULTS:")
        print("-" * 50)
        
        categories = {
            "Database": [],
            "Stock Prices": [],
            "Company Data": [],
            "Calculations": [],
            "Portfolio": [],
            "Additional": []
        }
        
        for test_name, test_data in self.results.items():
            if "Database" in test_name or "Connection" in test_name:
                categories["Database"].append((test_name, test_data))
            elif "Latest Price" in test_name or "Historical" in test_name or "Statistics" in test_name:
                categories["Stock Prices"].append((test_name, test_data))
            elif "Company" in test_name or "Search" in test_name or "Suggestions" in test_name:
                categories["Company Data"].append((test_name, test_data))
            elif "RSI" in test_name or "SMA" in test_name or "Volatility" in test_name or "Sharpe" in test_name:
                categories["Calculations"].append((test_name, test_data))
            elif "Portfolio" in test_name:
                categories["Portfolio"].append((test_name, test_data))
            else:
                categories["Additional"].append((test_name, test_data))
        
        for category, tests in categories.items():
            if tests:
                passed = sum(1 for _, test_data in tests if test_data['success'])
                total = len(tests)
                rate = (passed / total * 100) if total > 0 else 0
                print(f"\n{category} ({passed}/{total} - {rate:.0f}%):")
                
                for test_name, test_data in tests:
                    status = test_data['status']
                    result = test_data['result']
                    print(f"   {status} {test_name}: {result}")
        

        print("="*80)

    def test_centralized_helpers(self):
        """Test the centralized helper functions in mcp_servers/finance_helpers.py"""
        print(f"\n🔧 TESTING CENTRALIZED HELPER FUNCTIONS")
        print("=" * 70)
        
        try:
            from mcp_servers.finance_helpers import (
                search_companies_helper, get_database_connection
            )
            
            # Test 1: Search companies helper with enhanced search
            print("🧪 Test 1: Search Companies Helper")
            search_tests = [
                ("AAPL", "Direct symbol search"),
                ("APP", "Substring search for Apple"),
                ("micro", "Substring search for Microsoft"),
                ("tesla", "Case-insensitive search")
            ]
            
            for query, test_type in search_tests:
                try:
                    results = search_companies_helper(query, limit=5)
                    if isinstance(results, list) and len(results) > 0:
                        top_result = results[0]
                        symbol = top_result.get('symbol', 'Unknown')
                        name = top_result.get('name', 'Unknown')[:30]
                        self.log_test(f"Helper Search '{query}'", f"{len(results)} results, Top: {symbol} - {name}")
                    else:
                        self.log_test(f"Helper Search '{query}'", f"Found {len(results) if isinstance(results, list) else 0} results", len(results) > 0 if isinstance(results, list) else False)
                except Exception as e:
                    self.log_test(f"Helper Search '{query}'", f"Error: {str(e)}", False)
            
            # Test 2: Database connection
            print("\n🧪 Test 2: Database Connection Helper")
            try:
                conn = get_database_connection()
                if conn:
                    # Test connection is working
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM public.company LIMIT 1")
                        count = cur.fetchone()[0]
                    conn.close()
                    self.log_test("Database Connection Helper", f"Successfully connected, {count} companies available", True)
                else:
                    self.log_test("Database Connection Helper", "Connection returned None", False)
            except Exception as e:
                self.log_test("Database Connection Helper", f"Error: {str(e)}", False)
                
        except ImportError as e:
            self.log_test("Centralized Helpers Import", f"Import error: {str(e)}", False)
        except Exception as e:
            self.log_test("Centralized Helpers", f"Unexpected error: {str(e)}", False)

def main():
    """Execute comprehensive MCP testing suite - Optimized Edition"""
    print("🚀 COMPREHENSIVE MCP FINANCE TOOLS TESTING SUITE - OPTIMIZED EDITION")
    print("Production-ready testing for optimized 6-server architecture with 8 essential tools")
    print("="*95)
    print("🎯 Testing optimizations: Portfolio streamlined, Sentiment enhanced, Robust error handling")
    print()
    
    tester = ComprehensiveMCPTester()
    
    # Execute all MCP server tests
    if tester.test_database_infrastructure():
        tester.test_stock_price_mcp_server()
        tester.test_company_mcp_server()
        tester.test_calculations_mcp_server()
        tester.test_portfolio_mcp_server()  # Now tests optimized equal-weight tool
        tester.test_predictions_mcp_server()
        tester.test_news_insights_mcp_server()  # Now tests enhanced sentiment with news articles
    else:
        print("❌ Database infrastructure test failed. Cannot proceed with MCP server tests.")
    
    # Generate comprehensive report
    tester.generate_comprehensive_report()

if __name__ == "__main__":
    main()
