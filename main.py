from fastapi import FastAPI
import uvicorn
import contextlib

from mcp_servers import echo, math

app = FastAPI(
    title="Simple FastAPI Service with MCP",
)


# Mount multiple FastMCP servers into the application
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(echo.mcp.session_manager.run())
        await stack.enter_async_context(math.mcp.session_manager.run())
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/echo/", echo.mcp.streamable_http_app())
app.mount("/math/", math.mcp.streamable_http_app())


@app.get("/hello", operation_id="say_hello")
async def hello():
    """A simple greeting endpoint"""
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, log_level="debug")
