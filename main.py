from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
import sys
import os

app = FastAPI(title="Simple API")


@app.get("/hello", operation_id="say_hello")
async def hello():
    """A simple greeting endpoint"""
    return {"message": "Hello World"}


# Expose MCP server
mcp = FastApiMCP(app, name="Simple MCP Service")
mcp.mount()


def main():
    print("Hello from mcp-server-with-fastapi!")


def check():
    print(f"Python interpreter: {sys.executable}")
    print(f"Virtual environment active: {os.getenv('VIRTUAL_ENV') is not None}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
    main()
    check()
