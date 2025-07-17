from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
import contextlib

from mcp_servers import (
    echo,
    math,
    user_db,
    finance_db_company,
    finance_db_stock_price,
    finance_db_market_analysis,
    finance_analysis_and_predictions,
    finance_plotting,
    # finance_news_and_insights,
    finance_calculations,
    finance_portfolio,
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
            finance_db_market_analysis.mcp.session_manager.run()
        )
        await stack.enter_async_context(
            finance_analysis_and_predictions.mcp.session_manager.run()
        )
        await stack.enter_async_context(finance_plotting.mcp.session_manager.run())
        # await stack.enter_async_context(
        #     finance_news_and_insights.mcp.session_manager.run()
        # )
        await stack.enter_async_context(finance_calculations.mcp.session_manager.run())
        await stack.enter_async_context(finance_portfolio.mcp.session_manager.run())

        yield


app = FastAPI(
    title="Simple FastAPI with multiple FastMCP servers",
    lifespan=lifespan,
)
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
    "/finance_db_market_analysis/",
    finance_db_market_analysis.mcp.streamable_http_app(),
    name="finance_db_market_analysis",
)
app.mount(
    "/finance_analysis_and_predictions/",
    finance_analysis_and_predictions.mcp.streamable_http_app(),
    name="finance_analysis_and_predictions",
)
app.mount(
    "/finance_plotting/",
    finance_plotting.mcp.streamable_http_app(),
    name="finance_plotting",
)
# app.mount(
#     "/finance_news_and_insights/",
#     finance_news_and_insights.mcp.streamable_http_app(),
#     name="finance_news_and_insights",
# )

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


@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint that provides a welcome message.
    """
    return {"message": "Welcome to the FastAPI with multiple FastMCP servers!"}


if __name__ == "__main__":
    # uvicorn.run(
    #     "main:app",
    #     host=str(os.getenv("HOST", "localhost")),
    #     port=int(os.getenv("PORT", "8000")),
    #     log_level=str(os.getenv("LOG_LV", "info")),
    #     reload=os.getenv("RELOAD", "False") == "True",
    # )

    uvicorn.run("main:app")
