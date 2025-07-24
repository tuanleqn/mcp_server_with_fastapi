from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
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
    title="Finance MCP Server",
    description="FastAPI server with multiple finance MCP tools",
    lifespan=lifespan,
)

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

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
    """Root endpoint with basic information."""
    return {
        "message": "Finance MCP Server",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mcp_servers": [
            "echo", "math", "user_db", "finance_db_company", 
            "finance_db_stock_price", "finance_data_ingestion",
            "finance_calculations", "finance_portfolio", 
            "finance_plotting", "finance_news_and_insights",
            "finance_analysis_and_predictions"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=True
    )
