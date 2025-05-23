# Scryfall MCP Server

This project implements a Model Context Protocol (MCP) server that provides tools to interact with the [Scryfall API](https://scryfall.com/docs/api) for Magic: The Gathering card data.

## Description

The server listens for JSON-RPC requests over stdio and allows an MCP client to:
- Search for cards
- Retrieve card details by name, ID, set, etc.
- Get random cards
- Access Scryfall's card catalogs (e.g., artist names, card types)

## Prerequisites

- Python 3.x
- `requests` library

## Installation

1.  Clone this repository:
    ```bash
    git clone git@github.com:WilliamBCampbell/scryfall-mcp.git
    cd scryfall-mcp
    ```
2.  Install the required Python library:
    ```bash
    pip install requests
    ```
    (It's recommended to do this within a Python virtual environment.)

## Running the Server

This server is designed to be run by an MCP client. The client will typically execute the `scryfall_mcp_server.py` script.

The server communicates over stdin/stdout using JSON-RPC messages.

Refer to the `mcp-server.json` file for the server's capabilities and tool definitions.

## Logging

The server logs errors and informational messages to `mcp_server_stderr.log` in the same directory as the script.
