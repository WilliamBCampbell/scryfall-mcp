import sys
import os # For path joining

# Redirect stderr to a log file in the script's directory
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server_stderr.log")
try:
    # Open in append mode, create if not exists
    stderr_log_file = open(log_file_path, "a") 
    sys.stderr = stderr_log_file
except Exception as e:
    # If we can't open the log file, print to original stderr (which might go nowhere)
    print(f"CRITICAL: Could not open/redirect stderr to {log_file_path}. Error: {e}", file=sys.__stderr__) 

print("PYTHON SCRIPT: Starting up...", file=sys.stderr) # Execution trace
sys.stderr.flush() # Ensure it's written immediately

import json
import time # Keep time, requests will be imported below with error handling

try:
    print("PYTHON SCRIPT: Attempting to import 'requests'...", file=sys.stderr) # Execution trace
    sys.stderr.flush()
    import requests
    print("PYTHON SCRIPT: 'requests' imported successfully.", file=sys.stderr) # Execution trace
    sys.stderr.flush()
except ImportError as e:
    # Print to stderr so it doesn't interfere with MCP communication
    print(f"FATAL: Failed to import 'requests' library. Please ensure it is installed. Error: {e}", file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1) # Exit if requests is not available

print("PYTHON SCRIPT: Importing other modules...", file=sys.stderr) # Execution trace
sys.stderr.flush()
import logging
from typing import Dict, Optional, List, Any
from collections import deque
from urllib.parse import urljoin
# json and sys are already imported above
import argparse
# logging import is already above

class RateLimitedHTTPClient:
    """
    A robust HTTP client with sophisticated rate limiting capabilities.
    
    Features:
    - Configurable request rate limiting
    - Automatic retry mechanism
    - Logging of requests and errors
    - Sliding window rate limiting
    """
    
    def __init__(
        self, 
        base_url: str, 
        max_requests: int = 10,  # Max requests per interval (Scryfall: 10 requests/sec)
        interval: float = 1.0,   # Time interval in seconds (Scryfall: 1 second)
        timeout: float = 10.0,   # Default request timeout
        max_retries: int = 3     # Maximum number of retries for failed requests
    ):
        """
        Initialize the rate-limited HTTP client.
        
        Args:
            base_url (str): Base URL for all requests
            max_requests (int): Maximum number of requests per time interval
            interval (float): Time interval for rate limiting
            timeout (float): Request timeout in seconds
            max_retries (int): Number of times to retry a failed request
        """
        self.base_url = base_url
        self.max_requests = max_requests
        self.interval = interval
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._request_times = deque()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "ScryfallMCPServer/1.0"}) # Required by Scryfall
        
        self._logger = logging.getLogger(self.__class__.__name__)
        # BasicConfig should be called only once
        if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
            # Configure root logger if no handlers are present
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                stream=sys.stderr # Explicitly set stream to ensure it goes to our redirected stderr
            )
        # Ensure our specific logger also logs if basicConfig was already called elsewhere
        elif not self._logger.hasHandlers():
             self._logger.setLevel(logging.INFO) # Set level for this specific logger
             # If root logger already has handlers, this logger will use them by default.
             # If specific handling is needed (e.g. different format/level), add a handler here.


    def _apply_rate_limit(self):
        """
        Enforce rate limiting using a sliding window approach.
        """
        current_time = time.time()
        
        while self._request_times and current_time - self._request_times[0] > self.interval:
            self._request_times.popleft()
        
        if len(self._request_times) >= self.max_requests:
            oldest_request_time = self._request_times[0]
            wait_time = self.interval - (current_time - oldest_request_time)
            
            if wait_time > 0:
                self._logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time)
        
        self._request_times.append(time.time())

    def _log_request(self, method: str, url: str, params: Optional[Dict] = None):
        log_msg = f"{method} request to {url}"
        if params:
            log_msg += f" with params: {params}"
        self._logger.info(log_msg)

    def make_request(
        self, 
        endpoint: str, 
        method: str = 'GET', 
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        json_payload: Optional[Any] = None # Renamed from 'json' to avoid conflict
    ) -> requests.Response:
        full_url = urljoin(self.base_url, endpoint)
        self._apply_rate_limit()
        self._log_request(method, full_url, params)
        
        request_kwargs = {
            'timeout': self.timeout,
            'params': params or {},
            'headers': headers or {}
        }
        
        if json_payload is not None and method in ['POST', 'PUT', 'PATCH']:
            request_kwargs['json'] = json_payload
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.request(method, full_url, **request_kwargs)
                response.raise_for_status()
                return response
            
            except requests.RequestException as e:
                self._logger.error(f"Request failed (Attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt == self.max_retries:
                    self._logger.error("Max retries reached. Raising exception.")
                    raise
                wait_time = (2 ** attempt) * 0.1 
                self._logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        # Should not be reached if max_retries >= 0
        raise requests.RequestException("Request failed after all retries.")


    def close(self):
        self._session.close()

# --- MCP Server Tool Functions ---
SCRYFALL_API_BASE_URL = "https://api.scryfall.com"
print("PYTHON SCRIPT: Initializing RateLimitedHTTPClient...", file=sys.stderr) 
sys.stderr.flush()
client = RateLimitedHTTPClient(base_url=SCRYFALL_API_BASE_URL)
print("PYTHON SCRIPT: RateLimitedHTTPClient initialized.", file=sys.stderr) 
sys.stderr.flush()


def search_cards(query: str, unique: Optional[str] = "cards", order: Optional[str] = "name", 
                 direction: Optional[str] = "auto", include_extras: Optional[bool] = False, 
                 include_multilingual: Optional[bool] = False, include_variations: Optional[bool] = False, 
                 page: Optional[int] = 1) -> Dict:
    params = {
        "q": query, "unique": unique, "order": order, "dir": direction,
        "include_extras": include_extras, "include_multilingual": include_multilingual,
        "include_variations": include_variations, "page": page,
    }
    params = {k: v for k, v in params.items() if v is not None}
    logger = logging.getLogger("MCPStdioServer") # Ensure we use the main logger
    try:
        logger.info("search_cards: About to call client.make_request...")
        api_response_object = None # Initialize to None
        try:
            api_response_object = client.make_request("/cards/search", params=params)
        finally:
            logger.info(f"search_cards: client.make_request call finished. api_response_object is None: {api_response_object is None}")
            if api_response_object is not None:
                 logger.info(f"search_cards: Type of api_response_object post-call: {type(api_response_object)}")

        logger.info(f"search_cards: API request returned. Type of api_response_object: {type(api_response_object)}") 

        if api_response_object is None: # This check might be redundant if the finally block confirms it's not None
            logger.error("search_cards: client.make_request effectively returned None or did not assign.")
            return {"error": "Internal server error: API request failed to return a response object.", "details": "API client did not return a response.", "content": []}
        
        logger.info(f"search_cards: API response status code: {api_response_object.status_code}")
        
        scryfall_response = api_response_object.json()
        logger.info(f"search_cards: Raw Scryfall response: {json.dumps(scryfall_response)}")
        
        if scryfall_response.get("object") == "error":
            logger.warning(f"search_cards: Scryfall API returned an error object: {scryfall_response}")
            # Pass through Scryfall's error, but ensure "content" is present for Zod
            return {
                "error": scryfall_response.get("details", "Scryfall API error"),
                "scryfall_code": scryfall_response.get("code"),
                "scryfall_status": scryfall_response.get("status"),
                "content": [] 
            }
        
        # This block should only execute if not an error object
        content_data = scryfall_response.get("data", [])
        if not isinstance(content_data, list): # Ensure data is a list
            logger.warning(f"search_cards: Scryfall 'data' field was not a list: {content_data}. Using empty list.")
            content_data = []
        
        transformed_result = {
            "content": content_data,
            "total_cards": scryfall_response.get("total_cards"),
            "has_more": scryfall_response.get("has_more"),
            "next_page": scryfall_response.get("next_page")
        }
        
        # Clean up None values for keys that Zod might expect to be present or of a certain type
        # For example, if next_page is None, it's fine. If total_cards is None, it might be an issue
        # if Zod expects a number. Scryfall usually provides these.
        # For now, this explicit check for None and passing it is okay as JSON `null` is valid.
        for key in ["total_cards", "has_more", "next_page"]:
            if key not in transformed_result or transformed_result[key] is None:
                 # Scryfall might omit these if not applicable (e.g. no pagination)
                 # We can pass them as null or omit them. Zod schema on client side dictates.
                 # Assuming null is fine. If not, they should be omitted or given defaults.
                 pass # Keep as None (will be json null)

        logger.info(f"search_cards: Transformed result: {json.dumps(transformed_result)}")
        return transformed_result
    except requests.RequestException as e:
        logger.error(f"search_cards: RequestException: {str(e)}", exc_info=True)
        return {"error": str(e), "details": "Failed to search cards (RequestException).", "content": []}
    except json.JSONDecodeError as e:
        logger.error(f"search_cards: JSONDecodeError: {str(e)}", exc_info=True)
        return {"error": f"JSONDecodeError: {str(e)}", "details": "Failed to parse Scryfall response.", "content": []}
    except Exception as e:
        logger.error(f"search_cards: Unexpected error: {str(e)}", exc_info=True)
        return {"error": f"Unexpected error in search_cards: {str(e)}", "details": "An unexpected error occurred.", "content": []}


def get_card_by_name(name: str, search_type: str = "fuzzy", set_code: Optional[str] = None) -> Dict:
    params = {"exact": name} if search_type == "exact" else {"fuzzy": name}
    if set_code: params["set"] = set_code
    try:
        response = client.make_request("/cards/named", params=params)
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "details": f"Failed to get card by name: {name}"}

def get_card_by_tcgplayer_id(tcgplayer_id: int) -> Dict:
    try:
        response = client.make_request(f"/cards/tcgplayer/{tcgplayer_id}")
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "details": f"Failed to get card by TCGPlayer ID: {tcgplayer_id}"}

def get_cards_from_collection(identifiers: List[Dict[str, Any]]) -> Dict:
    payload = {"identifiers": identifiers}
    try:
        scryfall_response = client.make_request("/cards/collection", method="POST", json_payload=payload).json()

        if scryfall_response.get("object") == "error":
            return {
                "error": scryfall_response.get("details", "Scryfall API error on collection fetch."),
                "warnings": scryfall_response.get("warnings"),
                "content": []
            }

        content_data = scryfall_response.get("data", [])
        if not isinstance(content_data, list):
            content_data = []
            
        return {
            "content": content_data,
            "not_found": scryfall_response.get("not_found", []) 
        }
    except requests.RequestException as e:
        return {"error": str(e), "details": "Failed to get cards from collection (RequestException).", "content": []}
    except json.JSONDecodeError as e:
        return {"error": f"JSONDecodeError: {str(e)}", "details": "Failed to parse Scryfall response.", "content": []}

def get_random_card(q: Optional[str] = None, format: Optional[str] = "json", 
                    face: Optional[str] = None, version: Optional[str] = "large", 
                    pretty: Optional[bool] = False) -> Dict:
    params = {"q": q, "format": format, "face": face, "version": version, "pretty": pretty}
    params = {k: v for k, v in params.items() if v is not None}
    try:
        response = client.make_request("/cards/random", params=params)
        if format == "text": return {"text_content": response.text}
        if format == "image": return {"image_url": response.url, "content_type": response.headers.get('Content-Type')}
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "details": "Failed to get random card."}

def get_card_by_set_code_and_number(code: str, number: str, lang: Optional[str] = None, 
                                    format: Optional[str] = "json", face: Optional[str] = None, 
                                    version: Optional[str] = "large", pretty: Optional[bool] = False) -> Dict:
    endpoint = f"/cards/{code}/{number}" + (f"/{lang}" if lang else "")
    params = {"format": format, "face": face, "version": version, "pretty": pretty}
    params = {k: v for k, v in params.items() if v is not None}
    try:
        response = client.make_request(endpoint, params=params)
        if format == "text": return {"text_content": response.text}
        if format == "image": return {"image_url": response.url, "content_type": response.headers.get('Content-Type')}
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "details": f"Failed to get card by set {code} and number {number}."}

def get_card_by_arena_id(arena_id: int, format: Optional[str] = "json", face: Optional[str] = None, 
                         version: Optional[str] = "large", pretty: Optional[bool] = False) -> Dict:
    params = {"format": format, "face": face, "version": version, "pretty": pretty}
    params = {k: v for k, v in params.items() if v is not None}
    try:
        response = client.make_request(f"/cards/arena/{arena_id}", params=params)
        if format == "text": return {"text_content": response.text}
        if format == "image": return {"image_url": response.url, "content_type": response.headers.get('Content-Type')}
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "details": f"Failed to get card by Arena ID: {arena_id}."}

def get_card_by_scryfall_id(scryfall_id: str, format: Optional[str] = "json", face: Optional[str] = None, 
                            version: Optional[str] = "large", pretty: Optional[bool] = False) -> Dict:
    params = {"format": format, "face": face, "version": version, "pretty": pretty}
    params = {k: v for k, v in params.items() if v is not None}
    try:
        response = client.make_request(f"/cards/{scryfall_id}", params=params)
        if format == "text": return {"text_content": response.text}
        if format == "image": return {"image_url": response.url, "content_type": response.headers.get('Content-Type')}
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "details": f"Failed to get card by Scryfall ID: {scryfall_id}."}

def get_card_rulings_by_scryfall_id(scryfall_id: str, format: Optional[str] = "json", 
                                    pretty: Optional[bool] = False) -> Dict:
    params = {"format": format, "pretty": pretty}
    params = {k: v for k, v in params.items() if v is not None}
    try:
        scryfall_response = client.make_request(f"/cards/{scryfall_id}/rulings", params=params).json()
        
        if scryfall_response.get("object") == "error":
            return {
                "error": scryfall_response.get("details", "Scryfall API error on rulings fetch."),
                "content": []
            }

        content_data = scryfall_response.get("data", [])
        if not isinstance(content_data, list):
            content_data = []
            
        return {
            "content": content_data,
            "has_more": scryfall_response.get("has_more"), # Rulings can be paginated
            "next_page": scryfall_response.get("next_page")
        }
    except requests.RequestException as e:
        return {"error": str(e), "details": f"Failed to get rulings for Scryfall ID: {scryfall_id} (RequestException).", "content": []}
    except json.JSONDecodeError as e:
        return {"error": f"JSONDecodeError: {str(e)}", "details": "Failed to parse Scryfall response.", "content": []}


def parse_mana_cost(cost: str, format: Optional[str] = "json", pretty: Optional[bool] = False) -> Dict:
    params = {"cost": cost, "format": format, "pretty": pretty}
    params = {k: v for k, v in params.items() if v is not None}
    try:
        response = client.make_request("/symbology/parse-mana", params=params)
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "details": f"Failed to parse mana cost: {cost}."}

def _get_catalog_generic(catalog_name: str, format: Optional[str] = "json", pretty: Optional[bool] = False) -> Dict:
    params = {"format": format, "pretty": pretty}
    params = {k: v for k, v in params.items() if v is not None}
    try:
        response = client.make_request(f"/catalog/{catalog_name}", params=params)
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "details": f"Failed to get catalog: {catalog_name}."}

get_catalog_artist_names = lambda format="json", pretty=False: _get_catalog_generic("artist-names", format, pretty)
get_catalog_word_bank = lambda format="json", pretty=False: _get_catalog_generic("word-bank", format, pretty)
get_catalog_supertypes = lambda format="json", pretty=False: _get_catalog_generic("supertypes", format, pretty)
get_catalog_card_types = lambda format="json", pretty=False: _get_catalog_generic("card-types", format, pretty)
get_catalog_artifact_types = lambda format="json", pretty=False: _get_catalog_generic("artifact-types", format, pretty)
get_catalog_battle_types = lambda format="json", pretty=False: _get_catalog_generic("battle-types", format, pretty)
get_catalog_creature_types = lambda format="json", pretty=False: _get_catalog_generic("creature-types", format, pretty)
get_catalog_enchantment_types = lambda format="json", pretty=False: _get_catalog_generic("enchantment-types", format, pretty)
get_catalog_land_types = lambda format="json", pretty=False: _get_catalog_generic("land-types", format, pretty)
get_catalog_planeswalker_types = lambda format="json", pretty=False: _get_catalog_generic("planeswalker-types", format, pretty)
get_catalog_spell_types = lambda format="json", pretty=False: _get_catalog_generic("spell-types", format, pretty)
get_catalog_powers = lambda format="json", pretty=False: _get_catalog_generic("powers", format, pretty)
get_catalog_toughnesses = lambda format="json", pretty=False: _get_catalog_generic("toughnesses", format, pretty)
get_catalog_loyalties = lambda format="json", pretty=False: _get_catalog_generic("loyalties", format, pretty)
get_catalog_keyword_abilities = lambda format="json", pretty=False: _get_catalog_generic("keyword-abilities", format, pretty)
get_catalog_keyword_actions = lambda format="json", pretty=False: _get_catalog_generic("keyword-actions", format, pretty)
get_catalog_ability_words = lambda format="json", pretty=False: _get_catalog_generic("ability-words", format, pretty)
get_catalog_flavor_words = lambda format="json", pretty=False: _get_catalog_generic("flavor-words", format, pretty)

def initialize(**kwargs) -> Dict:
    logger = logging.getLogger("MCPStdioServer") # Use the same logger as mcp_stdio_server
    logger.info(f"PYTHON SCRIPT: initialize function CALLED. RAW KWARGS: {kwargs}")
    protocol_version_received = kwargs.get('protocolVersion')
    logger.info(f"PYTHON SCRIPT: initialize - protocolVersion from kwargs: {protocol_version_received}")
    return {
        "protocolVersion": protocol_version_received or "2025-03-26", 
        "serverInfo": {"name": "scryfall-mcp", "version": "1.0.0"},
        "capabilities": {}
    }

# --- MCP Server Main Logic ---
def mcp_stdio_server():
    logger = logging.getLogger("MCPStdioServer")
    if not logger.hasHandlers(): 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stderr)
    logger.info("Scryfall MCP Server started. Listening on stdin for JSON requests.")

    tool_functions = {
        "search_cards": search_cards, "get_card_by_name": get_card_by_name,
        "get_card_by_tcgplayer_id": get_card_by_tcgplayer_id, "get_cards_from_collection": get_cards_from_collection,
        "get_random_card": get_random_card, "get_card_by_set_code_and_number": get_card_by_set_code_and_number,
        "get_card_by_arena_id": get_card_by_arena_id, "get_card_by_scryfall_id": get_card_by_scryfall_id,
        "get_card_rulings_by_scryfall_id": get_card_rulings_by_scryfall_id, "parse_mana_cost": parse_mana_cost,
        "get_catalog_artist_names": get_catalog_artist_names, "get_catalog_word_bank": get_catalog_word_bank,
        "get_catalog_supertypes": get_catalog_supertypes, "get_catalog_card_types": get_catalog_card_types,
        "get_catalog_artifact_types": get_catalog_artifact_types, "get_catalog_battle_types": get_catalog_battle_types,
        "get_catalog_creature_types": get_catalog_creature_types, "get_catalog_enchantment_types": get_catalog_enchantment_types,
        "get_catalog_land_types": get_catalog_land_types, "get_catalog_planeswalker_types": get_catalog_planeswalker_types,
        "get_catalog_spell_types": get_catalog_spell_types, "get_catalog_powers": get_catalog_powers,
        "get_catalog_toughnesses": get_catalog_toughnesses, "get_catalog_loyalties": get_catalog_loyalties,
        "get_catalog_keyword_abilities": get_catalog_keyword_abilities, "get_catalog_keyword_actions": get_catalog_keyword_actions,
        "get_catalog_ability_words": get_catalog_ability_words, "get_catalog_flavor_words": get_catalog_flavor_words,
        "initialize": initialize,
    }

    for line in sys.stdin:
        line = line.strip()
        if not line: continue

        logger.debug(f"Received raw input: {line}")
        response_id = None
        try:
            request_data = json.loads(line)
            original_method = request_data.get("method")
            original_params = request_data.get("params", {})
            response_id = request_data.get("id")

            current_tool_name = original_method
            current_tool_args = original_params
            
            logger.info(f"Before 'tools/call' check. original_method: '{original_method}' (type: {type(original_method)})")
            is_tools_call_method = original_method == "tools/call"
            logger.info(f"Comparison (original_method == 'tools/call'): {is_tools_call_method}")

            if is_tools_call_method:
                nested_tool_name = original_params.get("name") 
                nested_tool_args = original_params.get("arguments")
                
                logger.info(f"Inside 'tools/call' handler. original_params: {original_params}")
                logger.info(f"Extracted nested_tool_name: '{nested_tool_name}' (type: {type(nested_tool_name)})")
                logger.info(f"Extracted nested_tool_args: '{nested_tool_args}' (type: {type(nested_tool_args)})")

                if nested_tool_name and isinstance(nested_tool_args, dict):
                    current_tool_name = nested_tool_name
                    current_tool_args = nested_tool_args
                    logger.info(f"Unwrapped 'tools/call': actual tool '{current_tool_name}', actual args '{current_tool_args}'")
                else:
                    logger.error(
                        f"Failed to unwrap 'tools/call' due to malformed parameters. "
                        f"original_params: {original_params}. "
                        f"Condition check: nested_tool_name ('{nested_tool_name}') is truthy? {bool(nested_tool_name)}. "
                        f"Condition check: nested_tool_args ('{nested_tool_args}') is dict? {isinstance(nested_tool_args, dict)}."
                    )
                    error_payload = {"code": -32602, "message": "Invalid params", "data": "Malformed 'tools/call' parameters. Expected 'name' (string) and 'arguments' (object)."}
                    response = {"jsonrpc": "2.0", "error": error_payload, "id": response_id}
                    logger.debug(f"Sending response for malformed 'tools/call': {json.dumps(response)}")
                    print(json.dumps(response), flush=True)
                    continue
            
            if not isinstance(current_tool_args, dict):
                logger.warning(f"Tool '{current_tool_name}' received non-dictionary params: {current_tool_args}. Using empty dict.")
                current_tool_args = {}

            logger.info(f"Processing tool: {current_tool_name} with args: {current_tool_args}")

            if current_tool_name in tool_functions:
                result = tool_functions[current_tool_name](**current_tool_args)
                response = {"jsonrpc": "2.0", "result": result, "id": response_id}
            else:
                error_payload = {"code": -32601, "message": "Method not found", "data": f"Tool '{current_tool_name}' is not supported."}
                response = {"jsonrpc": "2.0", "error": error_payload, "id": response_id}
                logger.error(f"Unknown tool: {current_tool_name}")
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e} for input: {line}")
            error_payload = {"code": -32700, "message": "Parse error", "data": str(e)}
            response = {"jsonrpc": "2.0", "error": error_payload, "id": None}
        except Exception as e:
            logger.error(f"Error processing request: {e} for input: {line}", exc_info=True)
            error_payload = {"code": -32603, "message": "Internal error", "data": str(e)}
            response = {"jsonrpc": "2.0", "error": error_payload, "id": response_id}

        logger.debug(f"Sending response: {json.dumps(response)}")
        print(json.dumps(response), flush=True)

    logger.info("Scryfall MCP Server stdin closed. Exiting.")

if __name__ == '__main__':
    def direct_test_main():
        # This function is for direct testing and is not called during MCP server operation.
        # It's kept for potential local debugging by the user.
        logging.basicConfig(level=logging.INFO, stream=sys.stderr) # Ensure logs go to stderr for direct test
        logger = logging.getLogger("DirectTest") # Use a specific logger for direct tests
        logger.info("--- Starting Direct Test Main ---")
        
        # Test search_cards
        logger.info("\n--- Testing search_cards ---")
        search_result = search_cards(query="lightning bolt", page=1, unique="prints")
        if not search_result.get("error"):
            for card_data in search_result.get('data', [])[:2]:
                logger.info(f"Found card: {card_data.get('name')} from set {card_data.get('set_name')}")
        else:
            logger.error(f"Error in search_cards: {search_result.get('error')}")
        # ... (other direct tests can be added here by the user if needed)
        
        client.close() # Ensure global client is closed after tests
        logger.info("--- Direct Test Main Finished ---")

    # Default to MCP server execution
    print("PYTHON SCRIPT: Calling mcp_stdio_server()...", file=sys.stderr) 
    sys.stderr.flush()
    mcp_stdio_server()
    print("PYTHON SCRIPT: mcp_stdio_server() exited.", file=sys.stderr) 
    sys.stderr.flush()
