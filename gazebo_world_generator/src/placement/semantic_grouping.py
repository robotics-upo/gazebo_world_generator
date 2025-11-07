"""
Semantic Grouping Module for Natural Object Placement

This module uses LLM reasoning to identify which objects should be grouped together
based on their functional relationships (e.g., chair with desk, lamp with workbench).
"""

import json, json5
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer
from gazebo_world_generator.src.config.settings import DEFAULT_MODEL
from gazebo_world_generator.src.prompts.manager import PromptManager

logger = logging.getLogger(__name__)


@dataclass
class SemanticGroup:
    """Represents a group of functionally related objects"""
    group_id: str
    primary_object: str  # The "anchor" object (e.g., 'desk')
    related_objects: List[str]  # Objects that should be near it (e.g., ['chair', 'lamp'])
    relationship_type: str  # e.g., 'workspace', 'seating_area', 'storage_zone'
    proximity: str  # 'adjacent', 'nearby', 'surrounding'
    spatial_hint: str  # Natural language hint for placement
    spatial_arrangement: Dict[str, str] = None  # e.g., {'chair': 'in_front', 'monitor': 'on_top'}

CONTEXT_LIMIT = 8192
SAFETY_MARGIN = 256 

SEMANTIC_GROUPING_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["group_id", "primary_object", "related_objects", "relationship_type"],
        "properties": {
            "group_id": {"type": "string"},
            "primary_object": {"type": "string"},
            "related_objects": {
                "type": "array",
                "items": {"type": "string"}
            },
            "relationship_type": {"type": "string"},
            "proximity": {"type": "string", "enum": ["adjacent", "nearby", "surrounding"]},
            "spatial_hint": {"type": "string"}
        }
    }
}


class SemanticGroupingEngine:
    """Engine for determining semantic relationships between objects using LLM reasoning"""

    def __init__(self, llm_interface):
        self.llm_interface = llm_interface
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
        
        # Initialize PromptManager
        if llm_interface and hasattr(llm_interface, 'prompt_manager'):
            self.prompt_manager = llm_interface.prompt_manager
        else:
            try:
                self.prompt_manager = PromptManager(
                    template_dir='prompts/',
                    default_version='v1',
                    enable_metrics=True
                )
            except Exception as e:
                logger.warning(f"Failed to initialize PromptManager in SemanticGroupingEngine: {e}")
                self.prompt_manager = None

    def identify_object_groups(self, objects_to_place: List[Dict], room_type: str, room_name: str, model_info_cache: Dict[str, Dict] = None) -> List[SemanticGroup]:
        """
        Use LLM to identify semantic groups with model awareness.

        Determines appropriate clearances using real model dimensions.

        Args:
            objects_to_place: List of objects with 'type' and 'count' keys
            room_type: Type of room (office, warehouse, living_room, etc.)
            room_name: Name of the room for context
            model_info_cache: Pre-resolved model information with dimensions

        Returns:
            List of SemanticGroup objects describing functional groupings
        """
        if not objects_to_place:
            return []

        # Count total objects and build summary
        object_summary = {}
        for obj in objects_to_place:
            obj_type = obj['type']
            count = obj.get('count', 1)
            object_summary[obj_type] = object_summary.get(obj_type, 0) + count

        # Build object list string WITH dimensions if available
        object_list = []
        for obj_type, count in object_summary.items():
            if model_info_cache and obj_type in model_info_cache:
                dims = model_info_cache[obj_type]['dimensions']
                model_name = model_info_cache[obj_type]['model_name']
                object_list.append(f"  • {obj_type}: {count} (model: {model_name}, size: {dims[0]:.2f}m × {dims[1]:.2f}m × {dims[2]:.2f}m)")
            else:
                object_list.append(f"  • {obj_type}: {count}")
        object_list_str = "\n".join(object_list)

        # Use PromptManager if available
        if not self.prompt_manager:
            logger.warning("No PromptManager available, cannot perform semantic grouping")
            return []

        try:
            prompt_content = self.prompt_manager.render(
                'semantic_grouping',
                room_type=room_type,
                room_name=room_name,
                object_list=object_list_str
            )
            messages = [{"role": "user", "content": prompt_content}]
        except Exception as e:
            logger.error(f"Failed to render semantic_grouping template: {e}")
            return []

        try:
            logger.info(f"Requesting semantic grouping analysis for {len(object_summary)} object types...")
            
            prompt_string = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            input_tokens = len(self.tokenizer.encode(prompt_string))

            available_output_tokens = CONTEXT_LIMIT - input_tokens - SAFETY_MARGIN

            # Ensure it's not negative
            max_tokens_for_output = max(0, available_output_tokens)

            response = self.llm_interface.query(messages, max_tokens=max_tokens_for_output)

            logger.debug(f"Semantic grouping response: {len(response)} characters")

            # Extract JSON from response
            groups_data = self._extract_json_from_response(response)

            if not groups_data:
                logger.warning(f"LLM did not return valid JSON for semantic grouping. Response: {response[:200]}")
                return []

            # Validate and convert to SemanticGroup objects
            if not isinstance(groups_data, list):
                logger.warning("LLM response is not a list. Using no groups.")
                return []

            groups = []
            for group_data in groups_data:
                try:
                    group = SemanticGroup(
                        group_id=group_data.get('group_id', f"group_{len(groups)}"),
                        primary_object=group_data['primary_object'],
                        related_objects=group_data.get('related_objects', []),
                        relationship_type=group_data.get('relationship_type', 'general'),
                        proximity=group_data.get('proximity', 'nearby'),
                        spatial_hint=group_data.get('spatial_hint', ''),
                        spatial_arrangement=group_data.get('spatial_arrangement', {})
                    )
                    groups.append(group)
                    logger.debug(f"Created group: {group.group_id} - {group.primary_object} with {len(group.related_objects)} related objects, arrangement: {group.spatial_arrangement}")
                except KeyError as e:
                    logger.warning(f"Skipping invalid group data (missing key: {e}): {group_data}")
                    continue

            logger.info(f"✅ Identified {len(groups)} semantic groups")
            return groups

        except Exception as e:
            logger.error(f"Failed to identify semantic groups: {e}")
            return []

    def _extract_json_from_response(self, response: str) -> Optional[List[Dict]]:
        """Extract JSON array from LLM response, handling markdown code blocks"""
        if not response:
            return None

        # Try direct JSON parsing
        try:
            data = json.loads(response.strip())
            return data
        except json.JSONDecodeError:
            pass

        # Try json5 for more lenient parsing
        try:
            data = json5.loads(response.strip())
            return data
        except:
            pass

        # Try extracting from markdown code blocks
        import re

        # Look for ```json ... ``` or ``` ... ```
        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'\[.*\]'  # Just look for array
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    # Try both json and json5
                    try:
                        data = json.loads(json_str.strip())
                    except:
                        data = json5.loads(json_str.strip())
                    return data
                except (json.JSONDecodeError, IndexError, Exception):
                    continue

        logger.debug(f"Could not extract JSON from response. Tried all patterns.")
        return None

    def expand_groups_to_instances(self, groups: List[SemanticGroup], objects_to_place: List[Dict]) -> List[Dict]:
        """
        Expand semantic groups to include instance counts and assign specific items to groups.

        Args:
            groups: List of semantic group definitions
            objects_to_place: Original object list with counts

        Returns:
            List of expanded groups with instance assignments
        """
        # Count available objects
        object_counts = {}
        for obj in objects_to_place:
            obj_type = obj['type']
            count = obj.get('count', 1)
            object_counts[obj_type] = object_counts.get(obj_type, 0) + count

        expanded_groups = []
        object_usage = {obj_type: 0 for obj_type in object_counts.keys()}

        for group in groups:
            # Check how many instances of this group we can create
            # Limited by the primary object count
            primary_available = object_counts.get(group.primary_object, 0)
            if primary_available == 0:
                logger.debug(f"Skipping group {group.group_id} - no {group.primary_object} available")
                continue

            # For each instance of the primary object, try to create a group
            for instance_idx in range(primary_available):
                instance_objects = [group.primary_object]
                object_usage[group.primary_object] += 1

                # Try to add ALL available related objects (not just one)
                for related_obj in group.related_objects:
                    available = object_counts.get(related_obj, 0) - object_usage.get(related_obj, 0)
                    for _ in range(available):
                        instance_objects.append(related_obj)
                        object_usage[related_obj] += 1

                expanded_groups.append({
                    'group_id': f"{group.group_id}_{instance_idx}",
                    'group_type': group.relationship_type,
                    'objects': instance_objects,
                    'proximity': group.proximity,
                    'spatial_hint': group.spatial_hint,
                    'primary_object': group.primary_object,
                    'spatial_arrangement': group.spatial_arrangement or {}
                })

        # Add ungrouped objects as single-item groups
        for obj_type, count in object_counts.items():
            used = object_usage.get(obj_type, 0)
            remaining = count - used
            for i in range(remaining):
                expanded_groups.append({
                    'group_id': f"standalone_{obj_type}_{i}",
                    'group_type': 'standalone',
                    'objects': [obj_type],
                    'proximity': 'independent',
                    'spatial_hint': f"Standalone {obj_type}",
                    'primary_object': obj_type
                })

        logger.debug(f"Expanded to {len(expanded_groups)} group instances")
        return expanded_groups
