# Jupyter Notebook Tool

A Python package providing atomic tools for langchain-based AI agents to manipulate Jupyter notebooks. Built on
nbclient/nbformat, it enables programmatic notebook creation, loading, and manipulation.

## Core Functionality

The system maintains notebook sessions using tokens. Each operation requires a valid token obtained from either:

- `create_notebook()`: Creates new empty notebook
- `load_notebook(url)`: Loads notebook from URL or file path

## Available Operations

With a valid token, the following operations are supported:

- `list_cells(token)`: Get ordered list of cell IDs
- `create_cell(token, source, cell_type, position)`: Add new cell
- `update_cell(token, id, source)`: Modify cell content
- `execute_cell(token, id)`: Run code cell and get output
- `delete_cell(token, id)`: Remove cell from notebook
- `get_notebook(token)`: Retrieve current notebook state

## Error Handling

The system provides specific error types for common failure cases:

- InvalidTokenError
- CellNotFoundError
- CellTypeError
- NotebookLoadError
- KernelError

## Worked Example

See docs/example.ipynb for an example of how to use the package.
