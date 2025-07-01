from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Echo MCP Server", version="0.0.1", stateless_http=True)


@mcp.tool(description="Echoes back the input string")
def echo(input_string: str) -> str:
    """
    Echoes back the input string.

    Args:
        input_string (str): The string to echo back.

    Returns:
        str: The echoed string.
    """
    return f"Echo: {input_string}"
