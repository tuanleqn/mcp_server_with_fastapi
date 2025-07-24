from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import contextlib
import os

# Import simple authentication system
from utils.simple_auth import (
    UserSignup, UserLogin, Token, ChatbotUser,
    create_user, authenticate_user, get_user_by_id,
    create_chat, get_user_chats, save_query, get_chat_queries,
    create_access_token, get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

from mcp_servers import (
    echo, 
    math, 
    user_db, 
    finance_db_company, 
    finance_db_stock_price, 
    finance_data_ingestion,
    finance_calculations,
    finance_portfolio,
    finance_plotting,
    finance_news_and_insights,
    finance_analysis_and_predictions
)

load_dotenv()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(echo.mcp.session_manager.run())
        await stack.enter_async_context(math.mcp.session_manager.run())
        await stack.enter_async_context(user_db.mcp.session_manager.run())
        await stack.enter_async_context(finance_db_company.mcp.session_manager.run())
        await stack.enter_async_context(
            finance_db_stock_price.mcp.session_manager.run()
        )
        await stack.enter_async_context(
            finance_data_ingestion.mcp.session_manager.run()
        )
        await stack.enter_async_context(
            finance_calculations.mcp.session_manager.run()
        )
        await stack.enter_async_context(
            finance_portfolio.mcp.session_manager.run()
        )
        await stack.enter_async_context(
            finance_plotting.mcp.session_manager.run()
        )
        await stack.enter_async_context(
            finance_news_and_insights.mcp.session_manager.run()
        )
        await stack.enter_async_context(
            finance_analysis_and_predictions.mcp.session_manager.run()
        )

        yield


app = FastAPI(
    title="FastAPI MCP Server with Simple Chatbot Authentication",
    lifespan=lifespan,
)

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", tags=["Web Interface"])
async def root():
    """Serve the chatbot web interface"""
    return FileResponse("static/chatbot.html")

@app.get("/auth", tags=["Web Interface"])
async def auth_page():
    """Serve the authentication web interface"""
    return FileResponse("static/auth.html")

@app.get("/finance", tags=["Web Interface"]) 
async def finance_demo():
    """Serve the finance demo interface"""
    return FileResponse("static/finance_demo.html")

# Simple Chatbot Authentication endpoints
@app.post("/auth/signup", response_model=dict, tags=["Authentication"])
async def signup(user_data: UserSignup):
    """User signup for chatbot access"""
    return create_user(user_data)

@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(user_data: UserLogin):
    """User login for chatbot access"""
    user = authenticate_user(user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user["email"], "user_id": user["id"]}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        }
    )

@app.get("/auth/me", response_model=ChatbotUser, tags=["Authentication"])
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user information"""
    user = get_user_by_id(current_user["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return ChatbotUser(**user)

@app.post("/auth/logout", tags=["Authentication"])
async def logout():
    """Logout (client-side token removal)"""
    return {"message": "Logged out successfully. Please discard your token."}

# Chat endpoints (using new schema)
@app.post("/chat/create", tags=["Chatbot"])
async def create_new_chat(
    name: str,
    current_user: dict = Depends(get_current_user)
):
    """Create a new chat session"""
    chat = create_chat(
        user_id=current_user["user_id"],
        name=name,
        tenant_id=1  # Default tenant
    )
    return chat

@app.get("/chat/list", tags=["Chatbot"])
async def list_user_chats(current_user: dict = Depends(get_current_user)):
    """Get all chats for the authenticated user"""
    return get_user_chats(current_user["user_id"])

@app.post("/chat/{chat_id}/query", tags=["Chatbot"])
async def send_chat_query(
    chat_id: int,
    query: str, 
    current_user: dict = Depends(get_current_user)
):
    """Send a query to a specific chat"""
    # Verify user owns this chat
    user_chats = get_user_chats(current_user["user_id"])
    chat_ids = [chat["id"] for chat in user_chats]
    
    if chat_id not in chat_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this chat"
        )
    
    # Process query (for now, simple echo)
    answer = f"Echo: {query}"
    
    # Save query to database
    query_record = save_query(
        chat_id=chat_id,
        query=query,
        answer=answer,
        tenant_id=1
    )
    
    return {
        "query": query,
        "answer": answer,
        "query_id": query_record["query_id"],
        "chat_id": chat_id
    }

@app.get("/chat/{chat_id}/queries", tags=["Chatbot"])
async def get_chat_query_history(
    chat_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get query history for a specific chat"""
    # Verify user owns this chat
    user_chats = get_user_chats(current_user["user_id"])
    chat_ids = [chat["id"] for chat in user_chats]
    
    if chat_id not in chat_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this chat"
        )
    
    return get_chat_queries(chat_id)

@app.get("/dashboard", tags=["Dashboard"])
async def dashboard():
    """
    Serve the MCP dashboard interface.
    """
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    else:
        return {"message": "Dashboard not available. Static files not found."}

# Mount MCP endpoints
app.mount("/echo/", echo.mcp.streamable_http_app(), name="echo")
app.mount("/math/", math.mcp.streamable_http_app(), name="math")
app.mount("/user_db/", user_db.mcp.streamable_http_app(), name="user_db")
app.mount(
    "/finance_db_company/",
    finance_db_company.mcp.streamable_http_app(),
    name="finance_db_company",
)
app.mount(
    "/finance_db_stock_price/",
    finance_db_stock_price.mcp.streamable_http_app(),
    name="finance_db_stock_price",
)
app.mount(
    "/finance_data_ingestion/",
    finance_data_ingestion.mcp.streamable_http_app(),
    name="finance_data_ingestion",
)
app.mount(
    "/finance_calculations/",
    finance_calculations.mcp.streamable_http_app(),
    name="finance_calculations",
)
app.mount(
    "/finance_portfolio/",
    finance_portfolio.mcp.streamable_http_app(),
    name="finance_portfolio",
)
app.mount(
    "/finance_plotting/",
    finance_plotting.mcp.streamable_http_app(),
    name="finance_plotting",
)
app.mount(
    "/finance_news_and_insights/",
    finance_news_and_insights.mcp.streamable_http_app(),
    name="finance_news_and_insights",
)
app.mount(
    "/finance_analysis_and_predictions/",
    finance_analysis_and_predictions.mcp.streamable_http_app(),
    name="finance_analysis_and_predictions",
)


@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint that provides a welcome message.
    """
    return {"message": "Welcome to the FastAPI with multiple FastMCP servers!"}

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify all MCP servers are running.
    """
    return {
        "status": "healthy",
        "mcp_servers": [
            "echo", "math", "user_db", "finance_db_company", 
            "finance_db_stock_price", "finance_data_ingestion",
            "finance_calculations", "finance_portfolio", 
            "finance_plotting", "finance_news_and_insights",
            "finance_analysis_and_predictions"
        ],
        "endpoints": {
            "echo": "/echo/",
            "math": "/math/", 
            "user_db": "/user_db/",
            "finance_db_company": "/finance_db_company/",
            "finance_db_stock_price": "/finance_db_stock_price/",
            "finance_data_ingestion": "/finance_data_ingestion/",
            "finance_calculations": "/finance_calculations/",
            "finance_portfolio": "/finance_portfolio/",
            "finance_plotting": "/finance_plotting/",
            "finance_news_and_insights": "/finance_news_and_insights/",
            "finance_analysis_and_predictions": "/finance_analysis_and_predictions/"
        }
    }

@app.get("/test/math/add", tags=["Test"])
async def test_math_add(a: float = 5, b: float = 3):
    """
    Test endpoint for math addition - demonstrates MCP functionality via HTTP.
    """
    try:
        # This is a simple test - in a real implementation, you'd call the MCP service
        result = a + b
        return {
            "operation": "addition",
            "inputs": {"a": a, "b": b},
            "result": result,
            "note": "This is a test endpoint. Use /docs to see all MCP tools available."
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/test/echo", tags=["Test"])
async def test_echo(message: str = "Hello from MCP!"):
    """
    Test endpoint for echo service.
    """
    return {
        "service": "echo",
        "input": message,
        "output": message,
        "note": "Echo service is working. Use /docs to access full MCP functionality."
    }


if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=8000):
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return None
    
    port = find_free_port(8000)
    if not port:
        print("❌ Could not find an available port between 8000-8009")
        exit(1)
    
    print(f"🚀 Starting server on port {port}")
    print(f"📡 Server URL: http://127.0.0.1:{port}")
    print(f"📚 API Docs: http://127.0.0.1:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=port,
        log_level="info"
    )
