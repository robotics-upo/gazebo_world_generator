"""
Smart Model Resolver
LLM-driven intelligent model selection and retrieval with a local-first strategy.
"""

import json
import os
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set

from gazebo_world_generator.src.config.settings import Config
from gazebo_world_generator.src.core.data_models import PlacementConstraint
from gazebo_world_generator.src.utils.cache import get_cache
from gazebo_world_generator.src.prompts.manager import PromptManager

logger = logging.getLogger(__name__)


class SmartModelResolver:
    """--- Intelligent model resolver with LLM integration and online search fallback. ---"""

    def __init__(self, llm_interface=None, online_db=None, enable_cache=True):
        self.llm = llm_interface
        self.online_db = online_db
        self.enable_cache = enable_cache

        # Initialize PromptManager if LLM is available
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
                logger.warning(f"Failed to initialize PromptManager in resolver: {e}")
                self.prompt_manager = None

        # Load object-specific guidance from JSON file
        self._object_guidance_map = self._load_object_guidance()

        # Use persistent cache for model resolution results
        if enable_cache:
            self.model_cache = get_cache(
                name="model_resolution",
                max_size=1000,
                default_ttl=86400.0,  # 24 hours
                persistent=True
            )
        else:
            # Fallback to simple dict if caching disabled
            self.model_cache = {}

        self._local_model_names_cache: Optional[Set[str]] = None
        self._model_name_to_dir_map: Dict[str, str] = {}


        online_status = "Enabled" if self.online_db else "Disabled"
        llm_status = "Enabled" if self.llm else "Disabled"
        logger.debug(f"SmartModelResolver initialized. Local models will be scanned on first use. "
                    f"Online search: {online_status}, LLM: {llm_status}")

    @property
    def local_models(self) -> Set[str]:
        """Lazily scans and caches local Gazebo models on first access."""
        if self._local_model_names_cache is None:
            # Try to load from persistent cache first
            if self.enable_cache and self._load_model_database():
                logger.info(
                    f"Loaded {len(self._local_model_names_cache)} models from cache"
                )
            else:
                logger.debug("Scanning local model directories...")
                self._scan_local_models()
                logger.debug(f"Found {len(self._local_model_names_cache)} unique local models.")

                # Save to persistent cache
                if self.enable_cache:
                    self._save_model_database()
        return self._local_model_names_cache

    def _get_model_database_cache(self):
        """Get or create the model database cache."""
        if not hasattr(self, '_model_db_cache'):
            self._model_db_cache = get_cache(
                name="model_database",
                max_size=1,  # Only one entry needed
                default_ttl=604800.0,  # 1 week
                persistent=True
            )
        return self._model_db_cache

    def _load_model_database(self) -> bool:
        """
        Load model database from persistent cache.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            cache = self._get_model_database_cache()
            cached_data = cache.get("model_scan_results")

            if cached_data:
                self._local_model_names_cache = cached_data['model_names']
                self._model_name_to_dir_map = cached_data['name_to_dir_map']
                return True
        except Exception as e:
            logger.warning(f"Failed to load model database from cache: {e}")

        return False

    def _save_model_database(self):
        """Save model database to persistent cache."""
        try:
            cache = self._get_model_database_cache()
            cache.set("model_scan_results", {
                'model_names': self._local_model_names_cache,
                'name_to_dir_map': self._model_name_to_dir_map
            })
            logger.debug("Saved model database to persistent cache")
        except Exception as e:
            logger.warning(f"Failed to save model database to cache: {e}")

    def _scan_local_models(self):
        """
        Scans Gazebo paths, distinguishes model names from directory names,
        and handles directories with spaces by creating symlinks.
        """
        self._local_model_names_cache = set()
        self._model_name_to_dir_map = {}
        
        gazebo_paths = set(Config.GAZEBO_MODEL_PATHS)
        if gazebo_model_path_env := os.environ.get('GAZEBO_MODEL_PATH'):
            gazebo_paths.update(p for p in gazebo_model_path_env.split(':') if p)

        for path_str in gazebo_paths:
            base_path = Path(path_str).expanduser()
            if not base_path.is_dir():
                continue

            for model_dir in base_path.iterdir():
                if not model_dir.is_dir() or not (model_dir / "model.sdf").is_file():
                    continue

                dir_name = model_dir.name
                current_dir_name = dir_name

                if ' ' in dir_name:
                    clean_name = dir_name.replace(' ', '_')
                    symlink_path = base_path / clean_name
                    if not symlink_path.exists():
                        try:
                            symlink_path.symlink_to(model_dir, target_is_directory=True)
                            logger.info(f"Created symlink for spaced directory: {clean_name} -> '{dir_name}'")
                            current_dir_name = clean_name 
                        except (OSError, PermissionError) as e:
                            logger.warning(f"Could not create symlink for '{dir_name}': {e}. Skipping model.")
                            continue
                    else:
                        current_dir_name = clean_name # Symlink already exists, use it

                official_model_name = self._get_model_name_from_config(model_dir) or current_dir_name
                
                self._local_model_names_cache.add(official_model_name)

                self._model_name_to_dir_map[official_model_name] = current_dir_name
                
                # This ensures URIs inside model.sdf are resolvable
                self._ensure_mesh_uri_compatibility(base_path, model_dir, current_dir_name)

    def _create_space_free_symlink(self, base_path: Path, dir_name: str, clean_name: str):
        """Create a symlink without spaces pointing to the original directory."""
        symlink_path = base_path / clean_name

        if not symlink_path.exists():
            try:
                symlink_path.symlink_to(dir_name)
                logger.info(f"Created symlink: {clean_name} -> {dir_name} (removing spaces from directory name)")
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not create symlink {clean_name} -> {dir_name}: {e}")

    def _ensure_mesh_uri_compatibility(self, base_path: Path, model_dir: Path, dir_name: str):
        """Ensures mesh URIs in model.sdf can be resolved by creating symlinks if needed."""
        try:
            sdf_content = (model_dir / "model.sdf").read_text()
            import re
            mesh_uris = re.findall(r'model://([^/\s<>"]+)/meshes/', sdf_content)

            for mesh_model_name in set(mesh_uris):
                # If the mesh references a different model name than the directory
                if mesh_model_name != dir_name and mesh_model_name.lower() != dir_name.lower():
                    symlink_path = base_path / mesh_model_name

                    # Create symlink if it doesn't exist
                    if not symlink_path.exists():
                        try:
                            symlink_path.symlink_to(dir_name)
                            logger.info(f"Created symlink: {mesh_model_name} -> {dir_name} (for mesh URI compatibility)")
                        except (OSError, PermissionError) as e:
                            logger.warning(f"Could not create symlink {mesh_model_name} -> {dir_name}: {e}")
        except Exception as e:
            logger.debug(f"Could not check mesh URIs for {model_dir.name}: {e}")

    def _get_model_name_from_config(self, model_dir: Path) -> Optional[str]:
        """Reads the model name from model.config file."""
        config_file = model_dir / "model.config"
        if not config_file.is_file():
            return None
        try:
            tree = ET.parse(config_file)
            name_elem = tree.getroot().find('name')
            if name_elem is not None and name_elem.text:
                return name_elem.text.strip()
        except Exception:
            return None # Fallback to directory name if config is malformed
        return None

    def get_directory_for_model(self, model_name: str) -> Optional[str]:
        """Gets the filesystem directory name corresponding to an official model name."""
        _ = self.local_models # Ensure scan has run
        return self._model_name_to_dir_map.get(model_name)

    def _cache_get(self, key: str):
        """Get from cache (works with both Cache objects and dicts)."""
        if isinstance(self.model_cache, dict):
            return self.model_cache.get(key)
        return self.model_cache.get(key)

    def _cache_set(self, key: str, value: any):
        """Set in cache (works with both Cache objects and dicts)."""
        if isinstance(self.model_cache, dict):
            self.model_cache[key] = value
        else:
            self.model_cache.set(key, value)

    def find_best_model(self, object_type: str, context: str = "") -> Optional[str]:
        """Finds the best model URI for a given object type and context using a multi-step strategy."""
        cache_key = f"{object_type.lower()}_{context}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if object_type.lower() in Config.ESSENTIAL_MODELS:
            result = Config.ESSENTIAL_MODELS[object_type.lower()]
            self._cache_set(cache_key, result)
            return result

        selected_model_name = None

        if self.llm:
            selected_model_name = self._find_local_model_by_llm(object_type, context)

        if not selected_model_name and self.online_db:
            logger.info(f"No suitable local model found for '{object_type}'. Searching online...")
            selected_model_name = self._find_online_model(object_type, context)

        if not selected_model_name:
            logger.warning("LLM-guided searches failed. Falling back to keyword-based local search.")
            selected_model_name = self._find_local_model_by_keyword(object_type)

        if selected_model_name:
            model_dir = self.get_directory_for_model(selected_model_name)
            if model_dir:
                final_uri = f"model://{model_dir}"
                self._cache_set(cache_key, final_uri)
                logger.info(f"Final selection for '{object_type}': {final_uri} (from model '{selected_model_name}')")
                return final_uri
            else:
                 logger.error(f"LLM selected '{selected_model_name}' but couldn't find its directory.")

        fallback_uri = self._get_fallback_model(object_type)
        if fallback_uri:
            logger.info(f"🔧 Using hardcoded fallback: {fallback_uri}")
            self._cache_set(cache_key, fallback_uri)
        else:
             logger.error(f"Complete failure: No model found for '{object_type}' after all strategies")
             self._cache_set(cache_key, None)

        return fallback_uri

    def _load_object_guidance(self) -> Dict[str, str]:
        """Load object-specific guidance from JSON file."""
        try:
            # Try to get path from PromptManager if available
            if self.prompt_manager:
                guidance_file = self.prompt_manager.template_dir / 'v1' / 'object_guidance.json'
            else:
                # Fallback to hardcoded path (should rarely happen)
                guidance_file = Path('prompts/v1/object_guidance.json')
            
            if guidance_file.exists():
                with open(guidance_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Object guidance file not found: {guidance_file}")
                return {}
        except Exception as e:
            logger.warning(f"Failed to load object guidance: {e}")
            return {}

    def _get_object_specific_guidance(self, object_type: str) -> str:
        """Provides object-specific guidance for the LLM to improve selection accuracy."""
        object_lower = object_type.lower()
        for key, guidance in self._object_guidance_map.items():
            if key in object_lower or object_lower in key:
                return guidance
        return ""  # No specific guidance for this object type

    def _find_local_model_by_llm(self, object_type: str, context: str) -> Optional[str]:
        """Uses LLM to pick the most appropriate local model from available candidates."""
        if not self.prompt_manager:
            logger.debug("No PromptManager available, skipping LLM-based local model selection")
            return None
            
        candidates = list(self.local_models)[:75]
        
        try:
            object_guidance = self._get_object_specific_guidance(object_type)
            prompt_content = self.prompt_manager.render(
                'model_selection',
                object_type=object_type,
                context=context,
                is_online_search=False,
                object_guidance=object_guidance,
                available_models=sorted(candidates)
            )
            messages = [{"role": "user", "content": prompt_content}]
            
            response_text = self.llm.query(messages).strip().replace('**', '').split(':')[0]

            if response_text and response_text.lower() != "none" and response_text in self.local_models:
                logger.info(f"🧠 LLM selected local model '{response_text}'.")
                return response_text
        except Exception as e:
            logger.warning(f"LLM local model selection failed: {e}")
        return None

    def _find_local_model_by_keyword(self, object_type: str) -> Optional[str]:
        """--- Weighted keyword matching for a smarter non-LLM fallback. ---"""
        scores = {}
        obj_lower = object_type.lower()
        synonyms = Config.CORE_SYNONYMS.get(obj_lower, [])

        for model_name in self.local_models:
            model_lower = model_name.lower().replace('_', ' ')
            score = 0
            if obj_lower == model_lower:
                score = 100 # Exact match
            elif obj_lower in model_lower:
                score = 50 # Partial match
            
            for syn in synonyms:
                if syn in model_lower:
                    score = max(score, 75) 
            
            if score > 0:
                scores[model_name] = score
        
        if not scores:
            return None
        
        best_model = max(scores, key=scores.get)
        logger.info(f"Found local model '{best_model}' via weighted keyword match (score: {scores[best_model]}).")
        return best_model

    def _find_online_model(self, object_type: str, context: str) -> Optional[str]:
        """Searches for, validates, and downloads an online model. Returns the model URI."""
        try:
            online_models = self.online_db.search_online_models(object_type)
            if not online_models: return None

            best_model_info = self._llm_validate_online_models(object_type, online_models[:10], context) if self.llm else online_models[0]

            if best_model_info:
                local_model_path = self.online_db.download_model(best_model_info)
                if local_model_path and local_model_path.is_dir():
                    model_name = local_model_path.name
                    logger.info(f"🌐 Download successful. Using model '{model_name}'")
                    self.local_models.add(model_name)
                    return f"model://{model_name}" # Returns the full URI
                else:
                    logger.error(f"Failed to download or verify model '{best_model_info['name']}'.")

        except Exception as e:
            logger.error(f"Online model search failed for '{object_type}': {e}")
        return None


    def _llm_validate_online_models(self, object_type: str, models: List[Dict], context: str) -> Optional[Dict]:
        """Uses LLM to pick the most appropriate model from a list of online search results."""
        if not models:
            return None
        
        if not self.prompt_manager:
            logger.debug("No PromptManager available, using top-ranked online search result")
            return models[0]

        model_list = [f"{m['name']}: {m.get('description', '')} [Source: {m.get('owner', m.get('repo_name', 'unknown'))}]" 
                      for m in models]
        
        try:
            object_guidance = self._get_object_specific_guidance(object_type)
            prompt_content = self.prompt_manager.render(
                'model_selection',
                object_type=object_type,
                context=context,
                is_online_search=True,
                object_guidance=object_guidance,
                available_models=model_list
            )
            messages = [{"role": "user", "content": prompt_content}]
            
            response = self.llm.query(messages).strip().replace('**', '').split(':')[0]

            if response and response.lower() != "none":
                for model in models:
                    if model['name'] == response:
                        logger.info(f"🧠 LLM validated online model: '{response}'")
                        return model
        except Exception as e:
            logger.warning(f"LLM online model validation failed: {e}")
            logger.warning("Falling back to top-ranked search result.")
            return models[0]
        
        logger.info("LLM validation concluded that no online models were suitable.")
        return None

    def _get_fallback_model(self, object_type: str) -> Optional[str]:
        """Provides a hardcoded fallback model for essential object types."""
        fallback = Config.FALLBACK_MODELS.get(object_type.lower())
        if fallback:
            logger.info(f"🔧 Using fallback model for '{object_type}': {fallback}")
        return fallback
