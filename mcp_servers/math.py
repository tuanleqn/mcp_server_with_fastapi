from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Math MCP Server", stateless_http=True)


@mcp.tool(description="Adds two numbers")
def add(a: float, b: float) -> float:
    """
    Adds two numbers.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The sum of the two numbers.
    """
    return a + b
