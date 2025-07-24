from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import contextlib
import os

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
    title="Simple FastAPI with multiple FastMCP servers",
    lifespan=lifespan,
)

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

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
