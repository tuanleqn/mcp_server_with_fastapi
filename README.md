# MCP Server with FastAPI

A demonstration of integrating FastAPI with Model Context Protocol (MCP) servers to create multiple API endpoints that can be used as tools for AI agents.

## Project Overview

This project implements a FastAPI application that mounts multiple MCP servers:
- **Echo Server**: A simple server that echoes back input strings
- **Math Server**: Provides basic mathematical operations
- **Database Server**: Connects to a PostgreSQL database to query user data
- **[Coming Soon]**: ...

## Prerequisites

- Python 3.12 or higher
- PostgreSQL database (for the DB server functionality)
- uv tools (for building the project)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp_server_with_fastapi
```

2. Create a virtual environment and activate it:
```bash
uv venv
```

3. Install dependencies:
```bash
uv sync
```

## Configuration

Create a `.env` file in the root directory with the following variables:
```
HOST=localhost
PORT=8000
LOG_LV=debug
DB_URI=postgresql://username:password@localhost:5432/yourdb
```

Adjust the values according to your environment.

## Running the Server

Execute the following command to start the server:
```bash
uv run main.py
```

The server will be accessible at `http://localhost:8000` (or the configured host/port).
