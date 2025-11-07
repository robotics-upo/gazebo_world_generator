"""
LLM Interface module for the Gazebo World Generator.

This module provides an abstraction layer for interacting with different Large Language Models (LLMs),
currently focusing on OpenAI-compatible models.
"""

import os
import re
import json
import logging
import time
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

import jsonschema
from openai import OpenAI, OpenAIError

from gazebo_world_generator.src.utils import llm_utils
from gazebo_world_generator.src.utils.circuit_breaker import get_circuit_breaker
from gazebo_world_generator.src.utils.cache import get_cache, cache_key
from gazebo_world_generator.src.utils.json_validator import JSONValidator
from gazebo_world_generator.src.exceptions import CircuitBreakerOpenError
from gazebo_world_generator.src.prompts.manager import PromptManager

logger = logging.getLogger(__name__)


ROOM_SCHEMA = {
    "type": "object",
    "properties": {
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "width": {"type": "number", "minimum": 2.0},
                            "length": {"type": "number", "minimum": 2.0},
                            "height": {"type": "number", "minimum": 2.0},
                        },
                    },
                    "connections": {
                        "type": "object",
                        "propertyNames": {"type": "string"},
                        "additionalProperties": {
                            "type": "string",
                            "enum": ["north", "east", "south", "west"],
                        },
                    },
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "count": {"type": "integer", "minimum": 1},
                                "semantic_context": {"type": "string"},
                            },
                            "required": ["type", "count"],
                        },
                    },
                },
                "required": ["name", "type"],
            },
        }
    },
    "required": ["rooms"],
}


class LLMBase(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def query(self, messages: List[Dict], max_tokens: int, temperature: float) -> str:
        """Send a query to the LLM and get a response."""
        pass


class OpenAICompatibleInterface(LLMBase):
    """Interface for communicating with OpenAI-compatible LLM servers."""

    def __init__(
        self, server_url: str, model_name: str = "default-model", timeout: float = 180.0,
        enable_circuit_breaker: bool = True,
        enable_cache: bool = True,
        cache_ttl: Optional[float] = 3600.0,  # 1 hour default
        prompt_version: str = "v1"  # Prompt template version
    ):
        base_url = f"http://{server_url}" if "://" not in server_url else server_url
        if not base_url.endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"

        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY", "sk-dummy-key")
        self.timeout = timeout
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl

        # Initialize PromptManager for template-based prompts
        try:
            self.prompt_manager = PromptManager(
                template_dir='prompts/',
                default_version=prompt_version,
                enable_metrics=True
            )
            logger.debug(f"PromptManager initialized with version '{prompt_version}'")
        except Exception as e:
            logger.warning(f"Failed to initialize PromptManager: {e}. Using fallback prompts.")
            self.prompt_manager = None

        # Initialize circuit breaker for resilience
        if enable_circuit_breaker:
            self.circuit_breaker = get_circuit_breaker(
                name=f"llm_{model_name}",
                failure_threshold=5,
                success_threshold=2,
                timeout=60.0,
                expected_exception=OpenAIError
            )
            logger.debug(f"Circuit breaker enabled for LLM '{model_name}'")
        else:
            self.circuit_breaker = None

        # Initialize cache for performance
        if enable_cache:
            self.cache = get_cache(
                name=f"llm_responses_{model_name}",
                max_size=500,  # Keep last 500 responses
                default_ttl=cache_ttl,
                persistent=True  # Survive restarts
            )
            logger.debug(f"Response cache enabled for LLM '{model_name}' (TTL={cache_ttl}s)")
        else:
            self.cache = None

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            # Log detailed info to file only
            logger.debug(
                f"LLM Interface initialized for model '{self.model_name}' at {base_url} with a {self.timeout}s timeout."
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def query(
        self, messages: List[Dict], max_tokens: int = 4096, temperature: float = 0.6
    ) -> str:
        """Send a query to the LLM with retry logic, caching, and circuit breaking."""
        if not self.client:
            return ""

        # Check cache first
        if self.cache:
            cache_k = cache_key(messages, max_tokens, temperature)
            cached_response = self.cache.get(cache_k)
            if cached_response is not None:
                logger.debug(f"Cache hit for LLM query (key={cache_k[:8]}...)")
                return cached_response

        # Calculate safe max_tokens based on model context limit
        MODEL_CONTEXT_LIMIT = 8192
        SAFETY_MARGIN = 200  # Reserve tokens for overhead and formatting
        
        # More conservative token estimation: ~3 chars per token (accounts for technical text, JSON)
        input_text = "\n".join([msg.get("content", "") for msg in messages])
        estimated_input_tokens = len(input_text) // 3
        
        # Calculate maximum available tokens for response
        available_tokens = MODEL_CONTEXT_LIMIT - estimated_input_tokens - SAFETY_MARGIN
        
        # Use the smaller of requested max_tokens or available space
        safe_max_tokens = min(max_tokens, available_tokens)
        
        # Ensure we always have at least some tokens for response
        if safe_max_tokens < 100:
            logger.warning(
                f"Input too large ({estimated_input_tokens} tokens). "
                f"Only {safe_max_tokens} tokens available for response."
            )
            safe_max_tokens = 100  # Minimum response size
        
        if safe_max_tokens < max_tokens:
            logger.debug(
                f"Adjusted max_tokens from {max_tokens} to {safe_max_tokens} "
                f"(input: ~{estimated_input_tokens} tokens, limit: {MODEL_CONTEXT_LIMIT})"
            )

        # Define the actual API call function
        def _make_api_call():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=safe_max_tokens,
                temperature=temperature,
                timeout=current_timeout,
            )

        retries = 3
        backoff_factor = 1.5
        current_timeout = self.timeout

        for attempt in range(retries):
            try:
                # Use circuit breaker if enabled, otherwise call directly
                if self.circuit_breaker:
                    try:
                        response = self.circuit_breaker.call(_make_api_call)
                    except CircuitBreakerOpenError as cb_error:
                        logger.error(f"Circuit breaker is open: {cb_error}")
                        # Return empty string to indicate failure without retrying
                        return ""
                else:
                    response = _make_api_call()

                # Extract response content
                content = response.choices[0].message.content

                # Cache the successful response
                if self.cache:
                    self.cache.set(cache_k, content, ttl=self.cache_ttl)
                    logger.debug(f"Cached LLM response (key={cache_k[:8]}...)")

                return content
            except OpenAIError as e:
                logger.error(f"LLM API error on attempt {attempt + 1}: {e}")
                if "timed out" in str(e).lower() and attempt < retries - 1:
                    logger.warning(
                        "Request timed out. Increasing timeout for the next attempt..."
                    )
                    current_timeout *= 1.5
                else:
                    current_timeout = self.timeout

                if attempt < retries - 1:
                    sleep_time = backoff_factor**attempt
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("LLM query failed after multiple retries.")
                    return ""
        return ""

    def parse_room_description(self, description: str, model_db) -> Dict:
        """Parse a high-level room description into a detailed JSON layout using the LLM."""
        logger.info("Parsing room description with LLM...")

        # Extract explicit dimensions from description
        dimension_hints = self._extract_dimension_hints(description)

        # Use PromptManager if available, otherwise fall back to hardcoded templates
        if self.prompt_manager:
            try:
                # Prepare context for template
                context = {}
                if dimension_hints:
                    hints_str = "USER-SPECIFIED DIMENSIONS (MUST USE THESE EXACTLY):\n"
                    for hint in dimension_hints:
                        hints_str += f"- {hint['room_name']}: {hint['width']}m × {hint['length']}m"
                        if hint.get('height'):
                            hints_str += f" × {hint['height']}m"
                        hints_str += "\n"
                    context['dimension_hints'] = hints_str
                    logger.info(f"Found {len(dimension_hints)} explicit dimension specification(s) in description")
                
                # Render prompt from template
                prompt_content = self.prompt_manager.render(
                    'room_parsing',
                    description=description,
                    context=context.get('dimension_hints', '')
                )
                
                messages = [
                    {"role": "user", "content": prompt_content}
                ]
            except Exception as e:
                logger.error(f"PromptManager failed to render room_parsing template: {e}")
                return self._create_fallback_room(description)
        else:
            # PromptManager not initialized - cannot proceed
            logger.error("PromptManager not initialized. Cannot parse room description.")
            return self._create_fallback_room(description)

        # Self-healing retry loop: try up to 3 times with correction prompts
        validator = JSONValidator(required_keys=["rooms"])
        max_retries = 3
        
        start_time = time.time()

        for attempt in range(max_retries):
            response_text = self.query(messages, temperature=0.2)

            if not response_text:
                logger.warning(f"LLM response was empty (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    messages.append({"role": "assistant", "content": "(empty response)"})
                    messages.append({"role": "user", "content": "Please provide a valid JSON response with the room layout."})
                    continue
                else:
                    logger.warning("All retry attempts exhausted. Falling back to manual room creation.")
                    return self._create_fallback_room(description)

            # Try to extract JSON first (LLM might add text before/after)
            try:
                parsed_json = llm_utils.extract_json_from_response(response_text)
                if not parsed_json:
                    raise ValueError("Failed to extract JSON from response")
            except Exception as e:
                logger.warning(f"JSON extraction failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": "Please provide a valid JSON response enclosed in ```json ``` code blocks or as a plain JSON object."})
                    continue
                else:
                    logger.error("Self-healing failed after all retry attempts. Falling back to manual room creation.")
                    return self._create_fallback_room(description)

            # Now validate the extracted JSON
            validation_result = validator.validate(json.dumps(parsed_json))

            if validation_result.is_valid:
                try:
                    # Use the already-parsed JSON

                    # Validate schema compliance
                    self._balance_object_distribution(parsed_json)
                    jsonschema.validate(instance=parsed_json, schema=ROOM_SCHEMA)

                    # Log parsed structure
                    for room in parsed_json.get('rooms', []):
                        room_name = room.get('name', 'Unknown')
                        objects = room.get('objects', [])
                        logger.debug(f"Room '{room_name}' objects: {objects}")
                        for obj in objects:
                            logger.info(f"  → Parsed object: type='{obj.get('type')}', count={obj.get('count', 1)}")

                    logger.info(f"✓ Successfully parsed {len(parsed_json.get('rooms', []))} rooms from description.")
                    
                    # Log prompt metrics if using PromptManager
                    if self.prompt_manager:
                        duration = time.time() - start_time
                        tokens = len(response_text) // 4  # Rough token estimate
                        self.prompt_manager.log_success(
                            template_name='room_parsing',
                            tokens_used=tokens,
                            response_time=duration
                        )
                    
                    return parsed_json

                except jsonschema.ValidationError as e:
                    logger.warning(f"JSON schema validation failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        # Add correction prompt
                        messages.append({"role": "assistant", "content": response_text})
                        correction_hint = validator.get_validation_hints([str(e)])
                        messages.append({"role": "user", "content": f"The JSON has a schema error: {str(e)}\n\n{correction_hint}\n\nPlease fix and provide a corrected JSON response."})
                        continue
            else:
                logger.warning(f"JSON validation failed (attempt {attempt + 1}/{max_retries}): {validation_result.error_summary}")
                if attempt < max_retries - 1:
                    # Add correction prompt with hints
                    messages.append({"role": "assistant", "content": response_text})
                    correction_hint = validator.get_validation_hints(validation_result.errors)
                    messages.append({"role": "user", "content": f"The response has errors:\n{validation_result.error_summary}\n\n{correction_hint}\n\nPlease fix and provide a corrected JSON response."})
                    continue

        # Log failure metrics if using PromptManager
        if self.prompt_manager:
            duration = time.time() - start_time
            self.prompt_manager.log_failure(
                template_name='room_parsing',
                error="Self-healing failed after all retry attempts"
            )
        
        logger.error("Self-healing failed after all retry attempts. Falling back to manual room creation.")
        return self._create_fallback_room(description)
    
    
    def _balance_object_distribution(self, parsed_data: Dict):
        """Enhanced programmatic sanity check to fix imbalanced object distribution from the LLM."""
        rooms = parsed_data.get("rooms", [])
        room_types = {}

        # Group rooms by type (excluding corridors and hallways)
        for i, room in enumerate(rooms):
            room_type = room.get("type")
            if room_type not in ["corridor", "hallway"]:
                if room_type not in room_types:
                    room_types[room_type] = []
                room_types[room_type].append(i)

        # Check for imbalanced distribution within each room type
        for room_type, indices in room_types.items():
            if len(indices) <= 1:
                continue  # Skip if only one room of this type

            # Count total instances per room (sum of 'count' field)
            all_objects = []
            total_instance_counts = []
            object_entry_counts = []

            for i in indices:
                room_objects = rooms[i].get("objects", [])
                all_objects.extend(room_objects)

                # Count total instances in this room
                total_instances = sum(obj.get('count', 1) for obj in room_objects)
                total_instance_counts.append(total_instances)
                object_entry_counts.append(len(room_objects))

            # Check if distribution is severely imbalanced
            if not all_objects:
                continue

            max_instances = max(total_instance_counts)
            min_instances = min(total_instance_counts)
            total_all_instances = sum(total_instance_counts)

            # If one room has all/most instances and others are nearly empty
            if total_all_instances > 0 and max_instances >= total_all_instances * 0.7 and min_instances == 0:
                logger.warning(f"Detected severe object imbalance in '{room_type}' rooms. Instance counts: {total_instance_counts}")
                logger.warning("Redistributing objects evenly across rooms...")

                # Flatten all objects by expanding 'count' into individual items
                expanded_objects = []
                for obj in all_objects:
                    # Skip if obj is not a dict (malformed data)
                    if not isinstance(obj, dict):
                        logger.warning(f"Skipping malformed object (not a dict): {obj}")
                        continue

                    count = obj.get('count', 1)
                    for _ in range(count):
                        expanded_objects.append({
                            'type': obj.get('type', 'unknown'),
                            'count': 1,
                            'semantic_context': obj.get('semantic_context', '')
                        })

                # Clear all objects from these rooms
                for i in indices:
                    rooms[i]["objects"] = []

                # Redistribute expanded objects evenly using round-robin
                for obj_idx, obj in enumerate(expanded_objects):
                    target_room_idx = indices[obj_idx % len(indices)]
                    rooms[target_room_idx].setdefault("objects", []).append(obj)

                # Log the new distribution
                new_instance_counts = [sum(obj.get('count', 1) for obj in rooms[i].get("objects", [])) for i in indices]
                logger.info(f"Rebalanced instance distribution: {new_instance_counts}")

            # Ensure empty rooms get at least one object
            elif min_instances == 0 and max_instances > 2:
                logger.warning(f"Found empty rooms in '{room_type}' type. Ensuring each room has at least one object.")

                empty_room_indices = [i for i in indices if sum(obj.get('count', 1) for obj in rooms[i].get("objects", [])) == 0]

                for empty_idx in empty_room_indices:
                    fullest_idx = max(indices, key=lambda i: sum(obj.get('count', 1) for obj in rooms[i].get("objects", [])))
                    fullest_objects = rooms[fullest_idx].get("objects", [])

                    if fullest_objects:
                        # Move one object from fullest to empty room
                        moved_obj = None
                        for obj in fullest_objects:
                            if obj.get('count', 1) > 1:
                                obj['count'] -= 1
                                moved_obj = {'type': obj['type'], 'count': 1, 'semantic_context': obj.get('semantic_context', '')}
                                break

                        if not moved_obj and fullest_objects:
                            moved_obj = fullest_objects.pop()

                        if moved_obj:
                            rooms[empty_idx].setdefault("objects", []).append(moved_obj)
                            logger.info(f"Moved 1 {moved_obj['type']} from '{rooms[fullest_idx]['name']}' to '{rooms[empty_idx]['name']}'")

        # Final validation: ensure corridors don't have objects
        for room in rooms:
            room_type = room.get("type", "").lower()

            # Clear any objects that might have been assigned to corridors by mistake
            if room_type in ["corridor", "hallway"]:
                if room.get("objects"):
                    logger.warning(f"Removing {len(room.get('objects', []))} objects incorrectly assigned to corridor '{room['name']}'")
                    room["objects"] = []
            # Allow empty rooms if user didn't specify objects for them
            elif not room.get("objects"):
                logger.info(f"Room '{room['name']}' has no objects (as requested by user)")

        logger.info("✅ Object distribution balance check completed.")


    def _extract_dimension_hints(self, description: str) -> List[Dict]:
        """
        Extract explicit room dimension specifications from user description.
        
        Supports patterns like:
        - "a 10m x 8m office"
        - "warehouse 20 meters by 15 meters"
        - "5m × 4m × 3m room"
        - "office (10x8m)"
        
        Returns:
            List of dimension hints with room name and dimensions
        """
        hints = []
        
        # Pattern 1: "WxL RoomType" format (e.g., "12m x 10m office", "10x8 warehouse")
        # Handles: "12m x 10m office", "10 x 8 office", "5mx4m room", etc.
        # Match dimensions (with optional 'm' attached to each number) followed by room type
        pattern1 = r'(\d+\.?\d*)m?\s*[x×X]\s*(\d+\.?\d*)m?(?:\s*[x×X]\s*(\d+\.?\d*)m?)?\s+(\w+)'
        matches1 = re.finditer(pattern1, description, re.IGNORECASE)
        for match in matches1:
            width = float(match.group(1))
            length = float(match.group(2))
            height = float(match.group(3)) if match.group(3) else None
            room_type = match.group(4).lower()
            
            # Filter: skip common connector/filler words that aren't room types
            skip_words = {'with', 'and', 'that', 'which', 'having', 'containing', 'of', 'in', 'at', 'on', 'by', 'the', 'a', 'an'}
            if room_type in skip_words:
                continue
            
            hint = {
                'room_name': room_type.capitalize(),
                'room_type': room_type,
                'width': width,
                'length': length
            }
            if height:
                hint['height'] = height
            hints.append(hint)
            logger.info(f"Found dimension hint (pattern1): {hint}")
        
        # Pattern 2: "RoomType (WxL)" or "RoomType (WxLxH)" (e.g., "office (10x8)")
        pattern2 = r'(\w+)\s*\((\d+\.?\d*)\s*[mx×xX]\s*(\d+\.?\d*)(?:\s*[mx×xX]\s*(\d+\.?\d*))?\s*(?:m|meters?)?\)'
        matches2 = re.finditer(pattern2, description, re.IGNORECASE)
        for match in matches2:
            room_type = match.group(1).lower()
            width = float(match.group(2))
            length = float(match.group(3))
            height = float(match.group(4)) if match.group(4) else None
            
            hint = {
                'room_name': room_type.capitalize(),
                'room_type': room_type,
                'width': width,
                'length': length
            }
            if height:
                hint['height'] = height
            hints.append(hint)
            logger.info(f"Found dimension hint (pattern2): {hint}")
        
        # Pattern 3: "RoomType of W meters by L meters" (e.g., "warehouse of 20 meters by 15 meters")
        pattern3 = r'(\w+)\s+of\s+(\d+\.?\d*)\s*(?:m|meters?)\s+by\s+(\d+\.?\d*)\s*(?:m|meters?)'
        matches3 = re.finditer(pattern3, description, re.IGNORECASE)
        for match in matches3:
            room_type = match.group(1).lower()
            width = float(match.group(2))
            length = float(match.group(3))
            
            hint = {
                'room_name': room_type.capitalize(),
                'room_type': room_type,
                'width': width,
                'length': length
            }
            hints.append(hint)
            logger.info(f"Found dimension hint (pattern3): {hint}")
        
        return hints

    def _create_fallback_room(self, description: str) -> Dict:
        """Create a simple fallback room layout if LLM parsing fails."""
        logger.warning("Using fallback room generation due to LLM failure or invalid response.")
        desc_lower = description.lower()
        room_type = "warehouse" if "warehouse" in desc_lower else "office"
        objects = [{"type": "desk", "count": 1, "semantic_context": "work surface"}]
        
        # Return a "sizeless" room. The generator is now responsible for sizing it.
        return { "rooms": [ { "name": "fallback_room", "type": room_type, "objects": objects } ] }
