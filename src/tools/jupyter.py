"""
Jupyter Notebook Manipulation Tools

This module provides a set of atomic tools for AI agents to manipulate Jupyter notebooks.
"""

import re
import nbclient
import nbformat
from nbformat import NotebookNode
from pydantic import BaseModel, ConfigDict, field_validator, field_serializer
import requests
from langchain.tools import tool
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import logging
from typing import Dict

# Configure logging
logger = logging.getLogger(__name__)

# Dictionary to store active notebook sessions
# Keys are unique tokens (UUIDs as strings) that identify each notebook session
# Values are tuples containing (NotebookClient for execution, NotebookNode for content)
_notebook_store: Dict[str, Tuple[NotebookClient, NotebookNode]] = {}

# Some hard coded version number
__version__: str = "0.1.0"

class NotebookError(Exception):
    """Base exception class for notebook operations."""
    pass

class InvalidTokenError(NotebookError):
    """Raised when a notebook token is invalid or not found."""
    pass

class CellNotFoundError(NotebookError):
    """Raised when a specified cell ID cannot be found in the notebook."""
    pass

class CellTypeError(NotebookError):
    """Raised when an operation is attempted on an incompatible cell type."""
    pass

class NotebookLoadError(NotebookError):
    """Raised when there is an error loading a notebook from a URL or file."""
    pass

class KernelError(NotebookError):
    """Raised when there is an error with the notebook kernel operations."""
    pass

class NotebookNodeAdapter(BaseModel):
    """Wraps the nbformat.NotebookNode class to allow for agent access using pydantic BaseModel.
    
    This adapter provides a Pydantic-compatible interface for working with nbformat.NotebookNode
    objects, enabling proper serialization and validation while maintaining compatibility with
    the notebook format.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    notebook: NotebookNode

    # Pre-validation: accept a NotebookNode instance or a dict.
    @field_validator("notebook", mode="before")
    def validate_notebook(cls, value):
        if isinstance(value, NotebookNode):
            return value
        if isinstance(value, dict):
            # Convert dict to NotebookNode.
            return NotebookNode(value)
        raise TypeError("notebook must be a NotebookNode or a dict")

    # Serialization: convert NotebookNode to a plain dict.
    @field_serializer("notebook")
    def serialize_notebook(self, notebook: NotebookNode) -> dict:
        return dict(notebook)

    
def _check_for_cell_ids(notebook: NotebookNode) -> NotebookNode:
    """Verifies that each cell in the notebook has a valid ID, adding new IDs where missing.
    
    The ID must match the regex pattern '^[a-zA-Z0-9-_]+$' and be no more than 64 characters.
    
    Args:
        notebook: The notebook to check and potentially modify
        
    Returns:
        The notebook with all cells having valid IDs
    """
    import re
    import uuid
    
    id_pattern = re.compile(r'^[a-zA-Z0-9-_]+$')
    
    for cell in notebook.cells:
        # Check if cell has a valid ID
        if not hasattr(cell, 'id') or not cell.id or not isinstance(cell.id, str) or \
           not id_pattern.match(cell.id) or len(cell.id) > 64 or len(cell.id) <1:
            # Generate a new UUID and remove any non-alphanumeric characters
            new_id = re.sub(r'[^a-zA-Z0-9-_]', '', str(uuid.uuid4()))[:64]
            cell.id = new_id
            logger.debug(f"Generated new cell ID: {new_id}")
            
    return notebook

def _lookup_notebook(token: str) -> Tuple[NotebookClient, NotebookNode]:
    """Look up a notebook adapter by its token.
    
    Args:
        token: The token used to store the notebook adapter
    
    Returns:
        A tuple of (NotebookClient,NotebookNode) from nbclient
    
    Raises:
        InvalidTokenError: If the token is not found in the store
    """
    if token not in _notebook_store:
        logger.warning(f"Notebook token not found: {token}")
        raise InvalidTokenError(f"Invalid notebook token: {token}")
    return _notebook_store[token]

@tool(parse_docstring=True, response_format="content_and_artifact")
def create_notebook() -> Tuple[str, NotebookNodeAdapter]:
    """Create a new empty Jupyter notebook and return a tuple containing a token
    for future use as well as a wrapped NotebookNode (read only).

    The token returned by the function should be used in subsequent calls to
    manipulate the notebook.

    Returns:
        A tuple where the first value is of type str and is the token that the caller
        should use for future manipulations to the notebook and the second value is
        of type NotebookNodeAdapter and may be used by the caller to inspect a read
        only copy of the notebook which has been created
    """
    try:
        logger.debug("Creating new empty notebook")
        # Create a new empty notebook using nbformat
        notebook = nbformat.v4.new_notebook()
        
        try:
            # Create a new client for working with this notebook and configure the kernel
            client = NotebookClient(notebook, kernel_name="python3")
            client.km = client.create_kernel_manager()
            client.start_new_kernel()
            client.kc = client.start_new_kernel_client()
        except Exception as e:
            raise KernelError(f"Failed to initialize notebook kernel: {str(e)}")
       
        # Store the NotebookClient and a token for future lookup
        token = str(uuid.uuid4())
        _notebook_store[token] = (client, notebook)
        logger.debug(f"Created new notebook token: {token}")

        # Return the token and adapter
        return token, NotebookNodeAdapter(notebook=notebook)

    except KernelError as e:
        # Re-raise known exceptions without wrapping
        logger.warning(str(e))
        raise
    except Exception as e:
        error_msg = f"Unexpected error creating notebook: {str(e)}"
        logger.warning(error_msg)
        raise NotebookError(error_msg) from e

@tool(parse_docstring=True, response_format="content_and_artifact")
def get_notebook(token: str) -> Optional[NotebookNodeAdapter]:
    """Retrieve a notebook from the store by its token.
    
    Args:
        token: The token used to store the notebook adapter
    
    Returns:
        A NotebookNodeAdapter containing the notebook if found, None otherwise
    """
    if token not in _notebook_store:
        logger.debug(f"Notebook token not found: {token}")
        return None
    _, notebook = _notebook_store[token]
    return NotebookNodeAdapter(notebook=notebook)

@tool(parse_docstring=True, response_format="content_and_artifact")
def load_notebook(url: str) -> Tuple[str,NotebookNodeAdapter]:
    """Loads a Jupyter notebook from a URL or local file path and returns a tuple
    containing a token for future used as well as a wrapped NotebookNode (read only).

    The token returned by the function should be used in subsequent calls to 
    manipulate the notebook.
    
    Args:
        url: The URL or file path of the notebook to load. Can be: (1) A remote URL (http:// or https://), (2) An absolute file path, (3) A relative file path (with or without ./)

    Returns:
        A tuple where the first value is of type str and is the token that the caller should use for future manipulations to the loaded notebook and the second value is of type NotebookNodeAdapter and may be used by the caller to inspect a read only copy of the notebook which has been loaded A copy of the loaded notebook
    """
    def try_load_url(url: str) -> str:
        """Attempt to load content from URL"""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text

    def try_load_file(path: str) -> str:
        """Attempt to load content from file"""
        # Remove ./ if present
        if path.startswith('./'):
            path = path[2:]
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def is_url(string: str) -> bool:
        """Check if string is a URL"""
        return string.startswith(('http://', 'https://'))

    try:
        logger.debug(f"Attempting to load notebook from: {url}")
        # First try loading as URL if it looks like one
        if is_url(url):
            try:
                notebook_content = try_load_url(url)
                logger.debug("Successfully loaded notebook from URL")
            except requests.RequestException as e:
                logger.warning(f"Failed to load from URL, trying as file: {str(e)}")
                try:
                    notebook_content = try_load_file(url)
                except (FileNotFoundError, PermissionError) as file_e:
                    raise NotebookLoadError(f"Failed to load from URL ({str(e)}) and file ({str(file_e)})")
        else:
            # Try as local file directly
            try:
                notebook_content = try_load_file(url)
                logger.debug("Successfully loaded notebook from file")
            except (FileNotFoundError, PermissionError) as e:
                # If local file fails and it might be a URL, try as URL
                if not url.startswith('/'):  # Don't try absolute paths as URLs
                    try:
                        logger.warning(f"Failed to load as file, trying as URL: {str(e)}")
                        notebook_content = try_load_url(url)
                    except requests.RequestException as url_e:
                        raise NotebookLoadError(f"Failed to load as both file ({str(e)}) and URL ({str(url_e)})")
                else:
                    raise NotebookLoadError(f"Failed to load notebook file: {str(e)}")

        try:
            # Load the notebook using nbformat
            notebook = nbformat.reads(notebook_content, as_version=4)
        except Exception as e:
            raise NotebookLoadError(f"Invalid notebook format: {str(e)}")
        
        # Ensure all cells have valid IDs
        notebook = _check_for_cell_ids(notebook)

        try:
            # Create a new client for working with this notebook and configure the kernel
            client = NotebookClient(notebook, kernel_name="python3")
            client.km = client.create_kernel_manager()
            client.start_new_kernel()
            client.kc = client.start_new_kernel_client()
        except Exception as e:
            raise KernelError(f"Failed to initialize notebook kernel: {str(e)}")
       
        # Store the NotebookClient and a token for future lookup
        token = str(uuid.uuid4())
        _notebook_store[token] = (client, notebook)
        logger.debug(f"Created new notebook token: {token}")

        # Return the token and adapter
        return token, NotebookNodeAdapter(notebook=notebook)

    except (NotebookLoadError, KernelError) as e:
        # Re-raise known exceptions without wrapping
        logger.warning(str(e))
        raise
    except Exception as e:
        error_msg = f"Unexpected error loading notebook from {url}: {str(e)}"
        logger.warning(error_msg)
        raise NotebookLoadError(error_msg) from e

@tool(parse_docstring=True)
def execute_cell(token: str, id: str) -> Dict[str, Any]:
    """Execute a specific cell in the notebook and return its results.
    
    Args:
        token: The token returned from :meth: load_notebook
        id: The ID of the cell to execute
    
    Returns:
        A dict of the execution results containing keys for output, execution_count, and status
    """
    try:
        logger.debug(f"Executing cell {id} for notebook {token}")
        # Get the NotebookClient and NotebookNode from the token
        client, notebook = _lookup_notebook(token)
        
        # Find and execute the specified cell. We need to keep track of the
        # cell position in the notebook because nbclient still uses cell index
        # to determine which cell should be executed.
        for index, cell in enumerate(notebook.cells):
            if cell.id == id:
                logger.debug(f"Attempting to execute cell {id} at index {index}")
                if cell.cell_type == "markdown":
                    logger.debug(f"Cell {id} is a markdown cell and cannot be executed")
                    raise CellTypeError(f"Cell {id} is a markdown cell and cannot be executed")
                try:
                    result = client.execute_cell(cell, index)
                    logger.debug(f"Cell {id} at index {index} executed successfully")
                    return {
                        "output": result.outputs,
                        "execution_count": result.execution_count,
                        "status": "ok" if not hasattr(result, "error") else "error"
                    }
                except CellExecutionError as e:
                    raise KernelError(f"Error executing cell {id}: {str(e)}") from e
                
        logger.warning(f"Cell with ID {id} not found")
        raise CellNotFoundError(f"Cell with ID {id} not found")
        
    except (InvalidTokenError, CellNotFoundError, CellTypeError, KernelError) as e:
        # Re-raise known exceptions without wrapping
        raise
    except Exception as e:
        error_msg = f"Unexpected error executing cell {id}: {str(e)}"
        logger.warning(error_msg)
        raise KernelError(error_msg) from e

@tool(parse_docstring=True)
def delete_cell(token: str, id: str) -> None:
    """Delete a specific cell from the notebook.
    
    Args:
        token: The token returned from load_notebook
        id: The ID of the cell to delete
    """
    try:
        logger.debug(f"Deleting cell {id} from notebook {token}")
        # Get the NotebookClient and NotebookNode from the token
        client, notebook = _lookup_notebook(token)
        
        # Find and delete the specified cell
        for idx, cell in enumerate(notebook.cells):
            if cell.id == id:
                del notebook.cells[idx]
                logger.debug(f"Cell {id} deleted successfully")
                return
                
        logger.warning(f"Cell with ID {id} not found")
        raise CellNotFoundError(f"Cell with ID {id} not found")
        
    except (InvalidTokenError, CellNotFoundError) as e:
        # Re-raise known exceptions without wrapping
        raise
    except Exception as e:
        error_msg = f"Unexpected error deleting cell {id}: {str(e)}"
        logger.warning(error_msg)
        raise NotebookError(error_msg) from e

@tool(parse_docstring=True)
def update_cell(token: str, id: str, source: str) -> None:
    """Update the content of a specific cell in the notebook.
    
    Args:
        token: The token returned from load_notebook
        id: The ID of the cell to update
        source: The new source content for the cell
    """
    try:
        logger.debug(f"Updating cell {id} in notebook {token}")
        # Get the NotebookClient and NotebookNode from the token
        client, notebook = _lookup_notebook(token)
        
        # Find and update the specified cell
        for cell in notebook.cells:
            if cell.id == id:
                cell.source = source
                logger.debug(f"Cell {id} updated successfully")
                return
                
        logger.warning(f"Cell with ID {id} not found")
        raise CellNotFoundError(f"Cell with ID {id} not found")
        
    except (InvalidTokenError, CellNotFoundError) as e:
        # Re-raise known exceptions without wrapping
        raise
    except Exception as e:
        error_msg = f"Unexpected error updating cell {id}: {str(e)}"
        logger.warning(error_msg)
        raise NotebookError(error_msg) from e

@tool(parse_docstring=True)
def list_cells(token: str) -> List[str]:
    """Get a list of all cell IDs in the notebook in order.
    
    Args:
        token: The token returned from load_notebook which will be a UUID
    
    Returns:
        A list of strings, each string being the ID of a cell in the notebook, in order

        
    Examples:
        1. Basic Usage:
           # Ensure you have captured a token from load_notebook
           token = load_notebook('some_file.ipynb')
           # Use the token variable to list all cells in the notebook
           result = list_cells(token)
           # Expected result: ['963ce5d5-725a-4c93-a29b-b31b9e76d605', 'b49d6d25-4322-45b0-821c-81a2e61dd84e']

        2. Handling an Invalid Token:
           token = "some-invalid-token"
           try:
               result = list_cells(token)
           except ValueError as e:
               # Expected: ValueError is raised with a descriptive error message.
               print(e)
    """
    try:
        logger.debug(f"Listing cells for notebook {token}")
        # Get the NotebookClient and NotebookNode from the token
        client, notebook = _lookup_notebook(token)
        
        # Collect all cell IDs in order
        cell_ids = [cell.id for cell in notebook.cells]
        logger.debug(f"Found {len(cell_ids)} cells")
        return cell_ids
        
    except InvalidTokenError:
        # Re-raise token errors without wrapping
        raise
    except Exception as e:
        error_msg = f"Unexpected error listing cells: {str(e)}"
        logger.warning(error_msg)
        raise NotebookError(error_msg) from e

@tool(parse_docstring=True)
def create_cell(token: str, source: str = "", cell_type: str = "code", position: int = -1) -> str:
    """Create a new cell in the notebook.
    
    Args:
        token: The token returned from load_notebook
        source: The source content for the new cell
        cell_type: The type of cell to create ("code" or "markdown")
        position: The position to insert the cell (-1 for end of notebook)
    
    Returns:
        The ID of the newly created cell
    """
    try:
        logger.debug(f"Creating new {cell_type} cell in notebook {token}")
        # Get the NotebookClient and NotebookNode from the token
        client, notebook = _lookup_notebook(token)
        
        # Create the new cell
        if cell_type == "code":
            cell = nbformat.v4.new_code_cell(source=source)
        elif cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source=source)
        else:
            logger.warning(f"Invalid cell type: {cell_type}")
            raise CellTypeError(f"Invalid cell type: {cell_type}. Must be 'code' or 'markdown'.")
            
        # Set the cell ID - ensure it matches the required pattern and length
        cell.id = re.sub(r'[^a-zA-Z0-9-_]', '', str(uuid.uuid4()))[:64]
        
        # Insert the cell at the specified position
        if position < 0:
            notebook.cells.append(cell)
        else:
            notebook.cells.insert(position, cell)
            
        logger.debug(f"Created new cell with ID: {cell.id}")
        return cell.id
        
    except (InvalidTokenError, CellTypeError) as e:
        # Re-raise known exceptions without wrapping
        raise
    except Exception as e:
        error_msg = f"Unexpected error creating cell: {str(e)}"
        logger.warning(error_msg)
        raise NotebookError(error_msg) from e
