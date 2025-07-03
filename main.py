import os
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
import contextlib

from mcp_servers import db, echo, math

load_dotenv()

app = FastAPI(
    title="Simple FastAPI Service with MCP",
)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(echo.mcp.session_manager.run())
        await stack.enter_async_context(math.mcp.session_manager.run())
        await stack.enter_async_context(db.mcp.session_manager.run())
        yield


# Mount multiple FastMCP servers into the application
app = FastAPI(lifespan=lifespan)
app.mount("/echo/", echo.mcp.streamable_http_app())
app.mount("/math/", math.mcp.streamable_http_app())
app.mount("/db/", db.mcp.streamable_http_app())


@app.get("/hello", operation_id="say_hello")
async def hello():
    """A simple greeting endpoint"""
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST"),
        port=int(os.getenv("PORT")),
        log_level=os.getenv("LOG_LV"),
    )
