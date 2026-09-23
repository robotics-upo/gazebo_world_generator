#!/usr/bin/env python3
"""
Refinement Planner
Parses natural language refinement requests using LLM.
"""

import logging
import json
import re
from typing import Dict

logger = logging.getLogger(__name__)


class RefinementPlanner:
    """
    Parses natural language refinement requests and converts them to structured operations.
    
    This class interacts with the LLM to understand user intent and generate
    structured refinement instructions.
    """
    
    def __init__(self, llm_interface, prompt_manager):
        """
        Initialize the planner with LLM interface and prompt manager.
        
        Args:
            llm_interface: LLM interface for querying the model
            prompt_manager: PromptManager for template rendering
        """
        self.llm = llm_interface
        self.prompt_manager = prompt_manager
    
    def parse_request(self, request: str, world_metadata: Dict, original_request: str = None) -> Dict:
        """
        Parse a natural language refinement request using LLM.
        
        Args:
            request: Natural language description of desired changes
            world_metadata: Current world metadata for context
            original_request: Original user request (for semantic context detection)
            
        Returns:
            dict: Structured refinement instructions
        """
        logger.info(f"Parsing refinement request: {request}")
        
        # Build context from current world
        context = self.build_context_summary(world_metadata)
        
        # Build detailed context
        obj_types = context.get('object_types', {})
        obj_summary = ', '.join([f"{count} {obj_type}(s)" for obj_type, count in obj_types.items()]) if obj_types else "None"
        room_count = context['rooms']
        room_names = context['room_names']
        
        # Add room requirement warning if multiple rooms
        room_requirement = ""
        if room_count > 1:
            room_requirement = f"\n\nIMPORTANT: This world has {room_count} rooms. You MUST specify which room in the 'properties.room' field for add/remove/modify/reposition operations. Available rooms: {', '.join(room_names)}"
        
        # Use PromptManager if available
        if not self.prompt_manager:
            logger.error("No PromptManager available, cannot parse refinement request")
            return {"error": "PromptManager not initialized"}
        
        try:
            prompt_content = self.prompt_manager.render(
                'world_refinement_parsing',
                room_count=room_count,
                room_names=room_names,
                total_objects=context['objects'],
                object_summary=obj_summary,
                room_requirement=room_requirement,
                user_request=request
            )
            
            # For refinement parsing, we use a single user message with all context
            messages = [{"role": "user", "content": prompt_content}]
        except Exception as e:
            logger.error(f"Failed to render world_refinement_parsing template: {e}")
            return {"error": f"Template rendering failed: {str(e)}"}
        
        try:
            response = self.llm.query(messages, temperature=0.2)
            logger.debug(f"LLM response: {response[:200]}...")
            
            # Extract JSON from response
            # Try multiple extraction methods
            refinement_data = None
            
            # Method 1: Direct JSON parse (if response is clean JSON)
            try:
                refinement_data = json.loads(response.strip())
            except json.JSONDecodeError:
                pass
            
            # Method 2: Extract from code blocks
            if not refinement_data:
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if code_block_match:
                    try:
                        refinement_data = json.loads(code_block_match.group(1))
                    except json.JSONDecodeError:
                        pass
            
            # Method 3: Find first complete JSON object
            if not refinement_data:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    try:
                        refinement_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            
            if refinement_data:
                # Validate required fields
                if "operation" not in refinement_data:
                    logger.warning("Missing 'operation' field in parsed JSON")
                    return {"error": "Invalid refinement format: missing operation"}
                
                logger.info(f"Successfully parsed refinement: {refinement_data.get('operation')}")
                return refinement_data
            else:
                logger.warning("Could not extract valid JSON from LLM response")
                logger.debug(f"Full response: {response}")
                return {"error": "Could not parse refinement request. Try rephrasing."}
                
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}", exc_info=True)
            return {"error": f"Parsing error: {str(e)}"}
    
    def build_context_summary(self, world_metadata: Dict) -> Dict:
        """
        Build a comprehensive summary of the current world for LLM context.
        
        Args:
            world_metadata: World metadata dictionary
            
        Returns:
            dict: Summarized context for LLM
        """
        return {
            'rooms': world_metadata.get('rooms', 0),
            'room_names': world_metadata.get('room_names', []),
            'objects': world_metadata.get('objects', 0),
            'object_types': world_metadata.get('object_types', {}),
            'object_names': world_metadata.get('object_names', [])[:15],
            'room_bounds': world_metadata.get('room_bounds', {})
        }
