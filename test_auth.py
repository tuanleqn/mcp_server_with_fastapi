#!/usr/bin/env python3
"""
Simple test script for chatbot authentication system
Tests the API endpoints to ensure everything is working correctly
"""

import requests
import json
import time
from datetime import datetime

# Server configuration
BASE_URL = "http://localhost:8000"
TEST_USER = "testuser"
TEST_PASSWORD = "testpassword"

def print_test_header(test_name):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*60}")

def print_result(success, message):
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {message}")

def test_server_health():
    """Test if server is running and healthy"""
    print_test_header("Server Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_result(True, f"Server is healthy - Status: {data.get('status', 'unknown')}")
            print(f"   MCP Servers: {data.get('mcp_servers', {})}")
            return True
        else:
            print_result(False, f"Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_result(False, f"Server not reachable: {e}")
        return False

def test_signup():
    """Test user signup functionality"""
    print_test_header("User Signup")
    
    # Create a unique test user
    test_email = f"test_{int(time.time())}@example.com"
    test_username = f"user_{int(time.time())}"
    
    signup_data = {
        "username": test_username,
        "email": test_email,
        "password": TEST_PASSWORD,
        "full_name": "Test User"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/signup", json=signup_data)
        if response.status_code == 201:
            data = response.json()
            print_result(True, f"User created successfully - ID: {data.get('user_id')}")
            return test_username, test_email
        else:
            print_result(False, f"Signup failed: {response.status_code} - {response.text}")
            return None, None
    except requests.exceptions.RequestException as e:
        print_result(False, f"Signup request failed: {e}")
        return None, None

def test_login(username=TEST_USER, password=TEST_PASSWORD):
    """Test user login and token generation"""
    print_test_header(f"User Login ({username})")
    
    login_data = {
        "username": username,
        "password": password
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        if response.status_code == 200:
            data = response.json()
            token = data.get('access_token')
            print_result(True, f"Login successful - Token type: {data.get('token_type')}")
            print(f"   Token preview: {token[:20]}...")
            return token
        else:
            print_result(False, f"Login failed: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print_result(False, f"Login request failed: {e}")
        return None

def test_auth_me(token):
    """Test getting current user info with token"""
    print_test_header("Get Current User Info")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print_result(True, f"Got user info - Username: {data.get('username')}")
            print(f"   Full name: {data.get('full_name')}")
            print(f"   Email: {data.get('email')}")
            print(f"   Created: {data.get('created_at')}")
            return True
        else:
            print_result(False, f"Get user info failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print_result(False, f"Get user info request failed: {e}")
        return False

def test_chat(token):
    """Test sending chat messages"""
    print_test_header("Chat Functionality")
    
    headers = {"Authorization": f"Bearer {token}"}
    test_messages = [
        "Hello chatbot!",
        "How are you today?",
        "What can you help me with?"
    ]
    
    success_count = 0
    
    for i, message in enumerate(test_messages, 1):
        chat_data = {"message": message}
        
        try:
            response = requests.post(f"{BASE_URL}/chat", json=chat_data, headers=headers)
            if response.status_code == 200:
                data = response.json()
                print_result(True, f"Chat {i}/3 - Message: '{message[:30]}...'")
                print(f"   Response: {data.get('response', 'No response')}")
                success_count += 1
            else:
                print_result(False, f"Chat {i}/3 failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print_result(False, f"Chat {i}/3 request failed: {e}")
    
    print(f"\n📊 Chat test summary: {success_count}/{len(test_messages)} messages successful")
    return success_count == len(test_messages)

def test_query(token):
    """Test query functionality"""
    print_test_header("Query Functionality")
    
    headers = {"Authorization": f"Bearer {token}"}
    test_queries = [
        {"query": "search companies", "query_type": "company_search"},
        {"query": "get stock info", "query_type": "stock_info"},
        {"query": "general question", "query_type": "general"}
    ]
    
    success_count = 0
    
    for i, query_data in enumerate(test_queries, 1):
        try:
            response = requests.post(f"{BASE_URL}/query", json=query_data, headers=headers)
            if response.status_code == 200:
                data = response.json()
                print_result(True, f"Query {i}/3 - Type: {query_data['query_type']}")
                print(f"   Query: {data.get('query')}")
                print(f"   Result: {data.get('result', 'No result')[:50]}...")
                success_count += 1
            else:
                print_result(False, f"Query {i}/3 failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print_result(False, f"Query {i}/3 request failed: {e}")
    
    print(f"\n📊 Query test summary: {success_count}/{len(test_queries)} queries successful")
    return success_count == len(test_queries)

def test_chat_history(token):
    """Test getting chat history"""
    print_test_header("Chat History")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/chat/history", headers=headers)
        if response.status_code == 200:
            data = response.json()
            chat_count = len(data.get('chats', []))
            print_result(True, f"Retrieved chat history - {chat_count} messages")
            
            if chat_count > 0:
                latest_chat = data['chats'][0]
                print(f"   Latest: {latest_chat.get('message', '')[:50]}...")
            return True
        else:
            print_result(False, f"Chat history failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_result(False, f"Chat history request failed: {e}")
        return False

def test_query_history(token):
    """Test getting query history"""
    print_test_header("Query History")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/query/history", headers=headers)
        if response.status_code == 200:
            data = response.json()
            query_count = len(data.get('queries', []))
            print_result(True, f"Retrieved query history - {query_count} queries")
            
            if query_count > 0:
                latest_query = data['queries'][0]
                print(f"   Latest: {latest_query.get('query', '')[:50]}...")
            return True
        else:
            print_result(False, f"Query history failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_result(False, f"Query history request failed: {e}")
        return False

def run_all_tests():
    """Run all authentication tests"""
    print("🚀 Starting Simple Chatbot Authentication Tests")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Server URL: {BASE_URL}")
    
    test_results = []
    
    # Test 1: Server health
    result = test_server_health()
    test_results.append(("Server Health", result))
    
    if not result:
        print("\n❌ Server is not running. Please start the server with 'python main.py'")
        return
    
    # Test 2: Login with existing test user
    token = test_login()
    test_results.append(("Login (existing user)", token is not None))
    
    if not token:
        print("\n❌ Cannot proceed without authentication token")
        return
    
    # Test 3: Get current user info
    result = test_auth_me(token)
    test_results.append(("Get User Info", result))
    
    # Test 4: Chat functionality
    result = test_chat(token)
    test_results.append(("Chat Messages", result))
    
    # Test 5: Query functionality
    result = test_query(token)
    test_results.append(("Query Functionality", result))
    
    # Test 6: Chat history
    result = test_chat_history(token)
    test_results.append(("Chat History", result))
    
    # Test 7: Query history
    result = test_query_history(token)
    test_results.append(("Query History", result))
    
    # Test 8: New user signup
    new_username, new_email = test_signup()
    test_results.append(("User Signup", new_username is not None))
    
    # Print final summary
    print_test_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n📊 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your chatbot authentication system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the server logs and database connection.")
    
    print(f"⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_all_tests()
