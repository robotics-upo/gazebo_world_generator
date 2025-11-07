"""
LLM utilities for reliably parsing, validating, and cleaning language model responses.
"""

import json
import logging
import re
from typing import Optional, List, Any, Dict
import math
import jsonschema 
import json5

logger = logging.getLogger(__name__)


def _extract_balanced_brackets(text: str, open_char: str, close_char: str) -> Optional[str]:
    """
    Extract a properly balanced bracket structure from text.
    Handles nested brackets correctly.
    
    Args:
        text: String starting with open_char
        open_char: Opening bracket character ('[' or '{')
        close_char: Closing bracket character (']' or '}')
    
    Returns:
        Extracted string with balanced brackets, or None if not found
    """
    if not text or text[0] != open_char:
        return None
    
    depth = 0
    for i, char in enumerate(text):
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[:i+1]
    
    return None  # No balanced structure found


def _fix_wrapped_json_strings(text: str) -> str:
    """
    Fix line breaks that occur inside JSON strings.
    The LLM API or terminal output sometimes wraps long strings, breaking JSON parsing.
    
    This function removes line breaks that appear within JSON string values.
    Strategy: Remove newlines that appear between quotes (inside string values).
    """
    result = []
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            # This character is escaped, just add it and continue
            result.append(char)
            escape_next = False
            continue
            
        if char == '\\':
            # Next character will be escaped
            result.append(char)
            escape_next = True
            continue
            
        if char == '"':
            # Toggle string state
            in_string = not in_string
            result.append(char)
            continue
            
        if char == '\n':
            if in_string:
                # We're inside a string, remove the newline
                result.append(' ')
            else:
                # We're outside a string, keep the newline (it's structural JSON)
                result.append(char)
        else:
            result.append(char)
    
    return ''.join(result)


def extract_json_from_response(response: str) -> Optional[Any]:
    """
    Master JSON Parser - Extracts, cleans, and parses JSON with robust error recovery.
    Enhanced with multiple extraction strategies and sanitization steps.
    """
    if not response:
        return None

    # Clean the response first
    response = response.strip()
    
    # Fix line breaks within JSON string values
    response = _fix_wrapped_json_strings(response)

    # Fix common LLM mistake: adding quotes between array elements
    response = re.sub(r'\}\},"\{', '}},{', response)
    response = re.sub(r'\}\}"\{', '}},{', response)
    response = re.sub(r'\},"\{', '},{', response)  # Also fix single brace version

    logger.debug(f"Attempting to extract JSON from response of length {len(response)}")

    # Strategy 1: Direct JSON parsing (fastest path)
    if (response.startswith('[') and response.endswith(']')) or (response.startswith('{') and response.endswith('}')):
        try:
            result = json5.loads(response)
            logger.debug(f"Strategy 1 (direct parsing) succeeded")
            return result
        except Exception as e:
            logger.debug(f"Strategy 1 (direct parsing) failed: {e}")
            pass

    # Strategy 2: Extract from markdown code blocks (with or without closing ```)
    # First, try to extract just the content between ```json and ``` (or ``` alone)
    markdown_patterns = [
        r'```json\s*\n?(.*?)\n?```',  # Non-greedy with ```json
        r'```\s*\n?(.*?)\n?```',      # Non-greedy with just ```
        r'```json\s*\n?(.*)',          # Without closing ``` (truncated)
        r'```\s*\n?(.*)',              # Without closing, no json marker
    ]
    
    for pattern in markdown_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                json_str = match.group(1).strip()
                # Log what we're trying to parse
                logger.debug(f"Attempting to parse markdown-extracted JSON of length {len(json_str)}")
                result = json5.loads(json_str)
                logger.debug(f"Strategy 2 (markdown extraction) succeeded with pattern: {pattern[:30]}")
                return result
            except Exception as e:
                logger.debug(f"Strategy 2 with pattern {pattern[:30]} failed: {e}")
                # Log the actual content that failed to help debugging
                logger.debug(f"Failed content preview: {json_str[:200] if len(json_str) > 200 else json_str}")
                continue

    # Strategy 3: Find largest JSON structure in response using GREEDY matching
    # This is crucial for arrays with multiple objects
    json_patterns = [
        r'\[[\s\S]*\]',  # Arrays (GREEDY - captures all content to last bracket)
        r'\{[\s\S]*\}'   # Objects (GREEDY)
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        # Try matches from longest to shortest
        for match_str in sorted(matches, key=len, reverse=True):
            try:
                # Try to sanitize and parse
                sanitized = _sanitize_json_string(match_str)
                result = json5.loads(sanitized)
                logger.debug(f"Successfully parsed JSON with pattern {pattern}, length: {len(match_str)}")
                return result
            except Exception as e:
                logger.debug(f"Failed to parse match of length {len(match_str)}: {e}")
                continue

    # Strategy 4: Emergency extraction with balanced bracket matching
    # Find properly balanced array structures
    array_start = response.find('[')
    if array_start != -1:
        try:
            array_str = _extract_balanced_brackets(response[array_start:], '[', ']')
            if array_str:
                sanitized = _sanitize_json_string(array_str)
                result = json5.loads(sanitized)
                logger.debug(f"Successfully parsed JSON using balanced bracket extraction")
                return result
        except Exception as e:
            logger.debug(f"Balanced bracket extraction failed: {e}")
            pass

    logger.error(f"All JSON extraction strategies failed. Response length: {len(response)}")
    logger.error(f"Response preview: '{response[:500]}...'")
    
    # Write full response to debug file for inspection
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', prefix='llm_parse_error_', suffix='.txt', delete=False) as f:
            f.write(response)
            logger.warning(f"Full response written to: {f.name}")
    except Exception as e:
        logger.debug(f"Could not write debug file: {e}")
    
    return None

def _sanitize_json_string(json_str: str) -> str:
    """Sanitize JSON string to fix common LLM errors."""
    # Remove leading/trailing whitespace
    json_str = json_str.strip()

    # Fix missing "x": key in pose objects (common LLM error)
    json_str = re.sub(r'("pose":\s*\{)\s*(-?[\d.]+)\s*,\s*("y":)',  r'\1"x": \2, \3', json_str)

    # Fix malformed extra closing braces with malformed key names
    json_str = re.sub(r'\}\s*,\s*["\']?bbox["\']?\s*:\s*\{[^}]*\}', '', json_str)  # Remove entire bbox objects
    json_str = re.sub(r'\}\s*,\s*["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?\s*:', r',"\1":', json_str)  # Fix },"key": patterns

    # Fix common number format issues
    json_str = re.sub(r':\s*\+(\d+)', r': \1', json_str)  # Remove leading + from numbers
    json_str = re.sub(r':\s*(\d+)\.0*\b', r': \1.0', json_str)  # Ensure .0 for floats

    # Fix missing quotes around string values
    json_str = re.sub(r'"type":\s*([a-zA-Z_][a-zA-Z0-9_]*)', r'"type": "\1"', json_str)

    # Fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    # Ensure consistent spacing
    json_str = re.sub(r'\s+', ' ', json_str)

    return json_str

def validate_json_with_schema(json_data: Any, schema: Dict) -> bool:
    """
    --- Validates a JSON object against a given jsonschema. ---

    Args:
        json_data: The parsed JSON data to validate.
        schema: The dictionary representing the JSON schema.

    Returns:
        True if the data is valid, False otherwise.
    """
    try:
        jsonschema.validate(instance=json_data, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        logger.warning(f"JSON schema validation failed: {e.message}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during schema validation: {e}")
        return False


def validate_coordinates(coords: Dict[str, float], bounds: Optional[Dict] = None) -> bool:
    """Validate that coordinates are within reasonable simulation bounds."""
    for key in ['x', 'y', 'z']:
        if key not in coords or not isinstance(coords[key], (int, float)):
            return False
        if not math.isfinite(coords[key]):
            return False

    if bounds:
        if not (bounds.get('x_min', -1000) <= coords['x'] <= bounds.get('x_max', 1000)): return False
        if not (bounds.get('y_min', -1000) <= coords['y'] <= bounds.get('y_max', 1000)): return False
        if not (bounds.get('z_min', -100)  <= coords['z'] <= bounds.get('z_max', 100)):  return False

    return True