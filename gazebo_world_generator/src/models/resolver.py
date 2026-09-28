"""
Smart Model Resolver
LLM-driven intelligent model selection and retrieval with a local-first strategy.
"""

import json
import os
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set

from gazebo_world_generator.src.config.settings import Config
from gazebo_world_generator.src.core.data_models import PlacementConstraint
from gazebo_world_generator.src.utils.cache import get_cache
from gazebo_world_generator.src.prompts.manager import PromptManager
from gazebo_world_generator.src.models.visual_quality import inspect_model_visuals
from gazebo_world_generator.src.resources import find_data_path

logger = logging.getLogger(__name__)

# A noun after the requested object often describes an accessory, not that object.
# For example, DeskPortrait is a portrait that sits on a desk.
ACCESSORY_HEADS = {
    "portrait", "picture", "photo", "sculpture", "toy", "lamp", "book",
    "clock", "monitor", "screen", "mug", "cup", "plant", "vase",
}
GROUND_FURNITURE = {
    "desk", "chair", "table", "bed", "sofa", "shelf", "bookshelf", "cabinet",
}
GENERIC_NAME_PARTS = {"unit", "system", "object", "item"}
NAME_ALIASES = {"shelving": "shelf", "shelves": "shelf", "racks": "rack"}


def _name_tokens(name: str) -> list[str]:
    return [part.lower() for part in re.findall(
        r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+", name.replace("_", " "))
            if not part.isdigit() and len(part) > 1]


class SmartModelResolver:
    """--- Intelligent model resolver with LLM integration and online search fallback. ---"""

    def __init__(self, llm_interface=None, online_db=None, enable_cache=True, search_paths=None):
        self.llm = llm_interface
        self.online_db = online_db
        self.enable_cache = enable_cache
        self.search_paths = search_paths

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
            cached_data = cache.get(self._scan_cache_key())

            if cached_data:
                self._local_model_names_cache = set(cached_data['model_names'])
                self._model_name_to_dir_map = cached_data['name_to_dir_map']
                return True
        except Exception as e:
            logger.warning(f"Failed to load model database from cache: {e}")

        return False

    def _save_model_database(self):
        """Save model database to persistent cache."""
        try:
            cache = self._get_model_database_cache()
            cache.set(self._scan_cache_key(), {
                'model_names': self._local_model_names_cache,
                'name_to_dir_map': self._model_name_to_dir_map
            })
            logger.debug("Saved model database to persistent cache")
        except Exception as e:
            logger.warning(f"Failed to save model database to cache: {e}")

    def _scan_cache_key(self):
        entries = []
        for directory in self._model_search_paths():
            mtime = directory.stat().st_mtime_ns if directory.exists() else 0
            entries.append(f"{directory}:{mtime}")
        return "model_scan_results:v2:" + ":".join(sorted(entries))

    def _model_search_paths(self):
        paths = set(str(path) for path in (self.search_paths or Config.GAZEBO_MODEL_PATHS))
        for variable in ("GAZEBO_MODEL_PATH", "GZ_SIM_RESOURCE_PATH"):
            paths.update(part for part in os.environ.get(variable, "").split(os.pathsep) if part)
        return sorted({Path(path).expanduser() for path in paths})

    def _scan_local_models(self):
        """
        Scans Gazebo paths, distinguishes model names from directory names,
        and handles directories with spaces by creating symlinks.
        """
        self._local_model_names_cache = set()
        self._model_name_to_dir_map = {}
        
        for base_path in self._model_search_paths():
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
                            # Relative, so the link still resolves when the directory
                            # is mounted elsewhere (e.g. a container's /data/models).
                            symlink_path.symlink_to(dir_name, target_is_directory=True)
                            logger.info(f"Created symlink for spaced directory: {clean_name} -> '{dir_name}'")
                            current_dir_name = clean_name 
                        except (OSError, PermissionError) as e:
                            logger.warning(f"Could not create symlink for '{dir_name}': {e}. Skipping model.")
                            continue
                    else:
                        current_dir_name = clean_name # Symlink already exists, use it

                official_model_name = self._get_model_name_from_config(model_dir) or current_dir_name
                
                self._local_model_names_cache.add(official_model_name)

                # Several Gazebo model directories may advertise the same
                # name in model.config. Prefer the directory whose name is
                # the advertised name, rather than whichever was scanned last.
                previous = self._model_name_to_dir_map.get(official_model_name)
                if previous is None or current_dir_name == official_model_name:
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
        if cached is not None and self._uri_suitable(cached, object_type):
            return cached
        if cached is not None:
            logger.warning("Discarding cached model %s for %s: visual or size check failed",
                           cached, object_type)

        if object_type.lower() in Config.ESSENTIAL_MODELS:
            result = Config.ESSENTIAL_MODELS[object_type.lower()]
            self._cache_set(cache_key, result)
            return result

        if self.llm:
            selected_model_name = self._find_local_model_by_llm(object_type, context)
            if selected_model_name and self._candidate_suitable(selected_model_name, object_type):
                return self._cache_selection(cache_key, selected_model_name, object_type)

        selected_model_name = self._find_local_model_by_keyword(object_type)
        if selected_model_name:
            return self._cache_selection(cache_key, selected_model_name, object_type)

        if self.online_db:
            logger.info(f"No suitable local model found for '{object_type}'. Searching online...")
            online_uri = self._find_online_model(object_type, context)
            if online_uri and self._uri_suitable(online_uri, object_type):
                self._cache_set(cache_key, online_uri)
                return online_uri

        fallback_uri = self._get_fallback_model(object_type)
        if fallback_uri and self._uri_suitable(fallback_uri, object_type):
            logger.info(f"🔧 Using hardcoded fallback: {fallback_uri}")
            self._cache_set(cache_key, fallback_uri)
            return fallback_uri
        logger.error("No visible, suitably sized model found for '%s'", object_type)
        self._cache_set(cache_key, None)
        return None

    def _cache_selection(self, cache_key: str, model_name: str, object_type: str) -> str:
        uri = f"model://{self.get_directory_for_model(model_name) or model_name}"
        self._cache_set(cache_key, uri)
        logger.info("Final selection for '%s': %s (from model '%s')", object_type, uri, model_name)
        return uri

    def _name_score(self, model_name: str, object_type: str) -> int:
        words = _name_tokens(model_name)
        requested = [NAME_ALIASES.get(word, word) for word in _name_tokens(object_type)
                     if word not in GENERIC_NAME_PARTS]
        normalized_words = [NAME_ALIASES.get(word, word) for word in words]
        if not requested:
            return 0
        if normalized_words == requested:
            return 100
        matches = [word for word in requested if word in normalized_words]
        if matches:
            last_index = max(normalized_words.index(word) for word in matches)
            if any(word in ACCESSORY_HEADS for word in words[last_index + 1:]):
                return -1
            if len(matches) == len(requested):
                return 80 if normalized_words[-1] == requested[-1] else 50
            return 35
        if any(synonym.replace("_", "") in model_name.lower().replace("_", "")
               for synonym in Config.CORE_SYNONYMS.get(object_type.lower(), [])):
            return 30
        return 0

    def _candidate_suitable(self, model_name: str, object_type: str) -> bool:
        if self._name_score(model_name, object_type) < 0:
            logger.warning("Rejected %s for %s: name describes an accessory", model_name, object_type)
            return False
        directory_name = self.get_directory_for_model(model_name) or model_name
        return self._uri_suitable(f"model://{directory_name}", object_type)

    def _uri_suitable(self, uri: str, object_type: str) -> bool:
        if not uri.startswith("model://"):
            return False
        directory_name = uri.removeprefix("model://")
        if self._name_score(directory_name, object_type) < 0:
            logger.warning("Rejected %s for %s: name describes an accessory", uri, object_type)
            return False
        model_dir = next((base / directory_name for base in self._model_search_paths()
                          if (base / directory_name / "model.sdf").is_file()), None)
        if model_dir is None:
            logger.warning("Rejected %s: model directory is unavailable", uri)
            return False
        inspection = inspect_model_visuals(model_dir, self._model_search_paths())
        if inspection.error:
            logger.warning("Rejected %s: %s", uri, inspection.error)
            return False
        expected = Config.DEFAULT_OBJECT_SIZES.get(object_type.lower())
        if object_type.lower() in GROUND_FURNITURE and expected and inspection.dimensions:
            actual_xy = sorted(inspection.dimensions[:2])
            expected_xy = sorted(expected[:2])
            if (actual_xy[0] < expected_xy[0] * 0.35 or
                    actual_xy[1] < expected_xy[1] * 0.35 or
                    inspection.dimensions[2] < expected[2] * 0.35):
                logger.warning("Rejected %s for %s: visual size %s is too small",
                               uri, object_type, inspection.dimensions)
                return False
        return True

    def _load_object_guidance(self) -> Dict[str, str]:
        """Load object-specific guidance from JSON file."""
        try:
            # Try to get path from PromptManager if available
            if self.prompt_manager:
                guidance_file = self.prompt_manager.template_dir / 'v1' / 'object_guidance.json'
            else:
                # Fallback to hardcoded path (should rarely happen)
                guidance_file = find_data_path("prompts/v1/object_guidance.json") or Path("prompts/v1/object_guidance.json")
            
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
            
        candidates = sorted(self.local_models,
                            key=lambda name: (-self._name_score(name, object_type), name))
        candidates = [name for name in candidates if self._name_score(name, object_type) > 0
                      and self._candidate_suitable(name, object_type)][:75]
        if not candidates:
            # Preserve semantic LLM selection when a model uses a related
            # name with no shared token (for example, couch for sofa).
            candidates = [name for name in sorted(self.local_models)
                          if self._candidate_suitable(name, object_type)][:75]
            if not candidates:
                return None
        
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

            if response_text and response_text.lower() != "none" and response_text in candidates:
                logger.info(f"🧠 LLM selected local model '{response_text}'.")
                return response_text
        except Exception as e:
            logger.warning(f"LLM local model selection failed: {e}")
        return None

    def _find_local_model_by_keyword(self, object_type: str) -> Optional[str]:
        """--- Weighted keyword matching for a smarter non-LLM fallback. ---"""
        candidates = sorted(self.local_models,
                            key=lambda name: (-self._name_score(name, object_type), name))
        for name in candidates:
            score = self._name_score(name, object_type)
            if score <= 0:
                break
            if self._candidate_suitable(name, object_type):
                logger.info("Found local model '%s' via keyword match (score: %s)", name, score)
                return name
        return None

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
                    self._model_name_to_dir_map[model_name] = model_name
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
