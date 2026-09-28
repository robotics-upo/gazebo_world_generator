"""
Online Model Database
Handles searching and downloading models from online Gazebo repositories.
"""

import logging
import os
import requests
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import zipfile
import tarfile
import tempfile
import shutil
import stat
import xml.etree.ElementTree as ET
import uuid
from urllib.parse import urlparse

from gazebo_world_generator.src.config.validated_settings import ModelsConfig

from gazebo_world_generator.src.config.settings import Config
from gazebo_world_generator.src.prompts.manager import PromptManager

logger = logging.getLogger(__name__)



class OnlineModelDatabase:
    """
    Interface for querying and downloading models from online Gazebo repositories.
    """

    def __init__(self, llm_interface=None, config: Optional[ModelsConfig] = None):
        self.llm = llm_interface
        self.config = config or ModelsConfig()
        self.search_cache: Dict[str, List[Dict]] = {}
        
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
                logger.warning(f"Failed to initialize PromptManager in OnlineModelDatabase: {e}")
                self.prompt_manager = None
        self.local_model_path = self.config.cache_directory.expanduser()
        self.local_model_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug("Online Model Database initialized with caching and parallel search.")
        logger.debug(f"Local model download path set to: {self.local_model_path}")

    def search_online_models(self, query: str, category: str = None) -> List[Dict]:
        """
        Comprehensive search across all online repositories using caching and parallel execution.
        """
        if query in self.search_cache:
            logger.info(f"⚡ Cache hit for '{query}'. Returning cached results.")
            return self.search_cache[query]

        search_terms = self._generate_search_terms(query)
        
        found_models = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_search = {
                executor.submit(self._search_gazebosim_models, search_terms): "Gazebo Fuel",
                executor.submit(self._search_github_models, search_terms): "GitHub"
            }
            logger.info("Executing online searches in parallel...")

            for future in as_completed(future_to_search):
                source = future_to_search[future]
                try:
                    models = future.result()
                    found_models.extend(models)
                except Exception as exc:
                    logger.error(f"{source} search generated an exception: {exc}")

        if not found_models:
            logger.warning(f"❌ No online models found for '{query}' after searching: {search_terms}")
            return []

        def sort_key(model):
            """
            Prioritization order:
            1. OpenRobotics models from Fuel (highest priority)
            2. Other Fuel models
            3. OSRF GitHub models
            4. Other GitHub models
            """
            score = 0
            source = model.get('source', '')
            owner = model.get('owner', '').lower()
            repo_name = model.get('repo_name', '').lower()

            # OpenRobotics Fuel models get highest priority
            if source == 'fuel' and owner == 'openrobotics':
                score = 1000
            # Other Fuel models
            elif source == 'fuel':
                score = 500
            # OSRF GitHub models (official Gazebo models)
            elif source == 'github' and 'osrf' in repo_name:
                score = 300
            # Other GitHub models
            elif source == 'github':
                score = 100

            # Add relevance score as tiebreaker
            score += model.get('relevance_score', 0)

            return score

        found_models.sort(key=sort_key, reverse=True)

        prioritized_names = [m['name'] for m in found_models[:5]]
        prioritized_sources = [f"{m['name']} ({m.get('owner', m.get('repo_name', 'unknown'))})" for m in found_models[:5]]
        logger.info(f"Top models after prioritization: {prioritized_sources}")

        ranked_models = self._rank_models_by_relevance(query, found_models)
        self.search_cache[query] = ranked_models
        return ranked_models

    def _make_request_with_retry(self, url: str, params: Optional[Dict] = None, **kwargs) -> Optional[requests.Response]:
        """Helper function to make HTTP requests with retry logic."""
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, timeout=15, **kwargs)
                if response.status_code in [429, 502, 503, 504]:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
        return None

    def _search_gazebosim_models(self, search_terms: List[str]) -> List[Dict]:
        """Searches the Gazebo Fuel repository with correct download URLs."""
        models = []
        fuel_api_url = "https://fuel.gazebosim.org/1.0/models"
        
        logger.debug(f"🔍 Searching Gazebo Fuel with terms: {search_terms}")
        for i, term in enumerate(search_terms):
            search_term = term.replace('_', '').replace('-', '')
            logger.debug(f"  ↪ Trying Fuel search term [{i+1}/{len(search_terms)}]: '{term}' (API query: '{search_term}')")
            response = self._make_request_with_retry(fuel_api_url, params={'q': search_term, 'per_page': 20})
            if response:
                results_count = len(response.json()) if isinstance(response.json(), list) else 0
                logger.debug(f"    ✓ Fuel returned {results_count} results for '{term}'")
                for model in response.json():
                    if isinstance(model, dict) and model.get('name') and model.get('owner'):
                        owner, name = model.get('owner'), model.get('name')
                        model_entry = {
                            'name': name, 'source': 'fuel', 'owner': owner,
                            'download_url': f"https://fuel.gazebosim.org/1.0/{owner}/models/{name}.zip",
                            'description': model.get('description', f'Gazebo model: {name} from Fuel'),
                            'relevance_score': 0.9 - (i * 0.1)
                        }
                        if not any(d['name'] == name and d['owner'] == owner for d in models):
                            models.append(model_entry)
        return models

    def _search_github_models(self, search_terms: List[str]) -> List[Dict]:
        """Searches GitHub repositories with correct download URLs."""
        models = []
        github_repos = [
            {'api_url': 'https://api.github.com/repos/osrf/gazebo_models/contents', 'name': 'OSRF Gazebo Models', 
             'zip_url': 'https://api.github.com/repos/osrf/gazebo_models/zipball/master'},
            {'api_url': 'https://api.github.com/repos/leonhartyao/gazebo_models_worlds_collection/contents/models', 'name': 'Gazebo Models Collection',
             'zip_url': 'https://api.github.com/repos/leonhartyao/gazebo_models_worlds_collection/zipball/master'}
        ]
        
        logger.debug(f"🔍 Searching GitHub repos with terms: {search_terms}")
        for repo in github_repos:
            logger.debug(f"  ↪ Searching {repo['name']}...")
            response = self._make_request_with_retry(repo['api_url'])
            if response:
                items = response.json() if isinstance(response.json(), list) else []
                logger.debug(f"    ✓ Found {len(items)} items in {repo['name']}")
                for item in items:
                    if item.get('type') == 'dir':
                        model_name = item['name']
                        model_name_normalized = model_name.lower().replace('_', ' ').replace('-', ' ')
                        for i, term in enumerate(search_terms):
                            term_normalized = term.lower().replace('_', ' ').replace('-', ' ')
                            # Check if term is in model name (fuzzy match)
                            if term_normalized in model_name_normalized or model_name_normalized in term_normalized:
                                model_entry = {
                                    'name': model_name, 'source': 'github', 'repo_name': repo['name'],
                                    'download_url': repo['zip_url'],
                                    'path_in_repo': item['path'], # Store the path to extract
                                    'description': f"Gazebo model from {repo['name']}", 
                                    'relevance_score': 0.8 - (i * 0.1)
                                }
                                if not any(d['name'] == model_name for d in models):
                                    models.append(model_entry)
                                break
        return models

    def _generate_search_terms(self, query: str) -> List[str]:
        """Use LLM with PromptManager or fallback to generate relevant search terms."""
        if self.llm and self.prompt_manager:
            try:
                prompt_content = self.prompt_manager.render('model_search_terms', query=query)
                messages = [{"role": "user", "content": prompt_content}]
                response = self.llm.query(messages, max_tokens=100)
                
                if response:
                    # Extract just the comma-separated terms from response
                    response_clean = response.strip().split('\n')[0]  # Take first line only
                    terms = [term.strip().lower().replace(' ', '_') for term in response_clean.split(',') if term.strip()]
                    if terms:
                        logger.info(f"🎯 LLM generated search terms: {terms[:4]}")
                        return terms[:4]
            except Exception as e:
                logger.warning(f"LLM search term generation failed: {e}")

        # Enhanced fallback logic with better synonym mappings
        query_lower = query.lower().replace(' ', '_')
        
        # Enhanced synonym dictionary for common furniture
        enhanced_synonyms = {
            'tv_stand': ['entertainment_center', 'media_console', 'tv_unit', 'tv_cabinet'],
            'nightstand': ['bedside_table', 'night_table', 'side_table', 'bedside_cabinet'],
            'coffee_table': ['living_room_table', 'center_table', 'low_table', 'lounge_table'],
            'dining_table': ['dinner_table', 'kitchen_table', 'eating_table', 'table'],
            'sofa': ['couch', 'settee', 'living_room_chair', 'lounge_seat'],
            'wardrobe': ['closet', 'armoire', 'clothes_cabinet', 'clothing_storage'],
            'dresser': ['chest_of_drawers', 'drawer_unit', 'bedroom_dresser', 'cabinet'],
            'armchair': ['accent_chair', 'lounge_chair', 'easy_chair', 'single_seat'],
            'bookshelf': ['bookcase', 'shelf', 'book_storage', 'shelving_unit'],
            'desk': ['writing_desk', 'office_desk', 'work_table', 'computer_desk'],
        }
        
        base_terms = [query_lower]
        # Try exact match first
        if query_lower in enhanced_synonyms:
            base_terms.extend(enhanced_synonyms[query_lower])
        else:
            # Try partial match (e.g., 'office_desk' matches 'desk')
            for key, synonyms in enhanced_synonyms.items():
                if key in query_lower or query_lower in key:
                    base_terms.extend(synonyms)
                    break
            else:
                # Fallback to config synonyms
                base_terms.extend(Config.CORE_SYNONYMS.get(query_lower, []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in base_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        logger.debug(f"Fallback search terms for '{query}': {unique_terms[:4]}")
        return unique_terms[:4]

    def _rank_models_by_relevance(self, original_query: str, models: List[Dict]) -> List[Dict]:
        """Use LLM to rank and filter found models by relevance."""
        if not models:
            return []
        
        if not self.llm or not self.prompt_manager:
            logger.debug("LLM or PromptManager not available, using relevance score sorting")
            return sorted(models, key=lambda x: x.get('relevance_score', 0), reverse=True)[:15]

        # Prepare models for ranking (limit to top 25 to avoid token overload)
        models_to_rank = models[:25]
        
        try:
            model_dicts = [
                {
                    'name': model['name'],
                    'description': model.get('description', 'No description')
                }
                for model in models_to_rank
            ]
            
            prompt_content = self.prompt_manager.render(
                'online_model_ranking',
                query=original_query,
                models=model_dicts
            )
            messages = [{"role": "user", "content": prompt_content}]
            
            response = self.llm.query(messages, max_tokens=500)
            
            if response:
                logger.debug(f"LLM ranking response ({len(response)} chars): {response[:200]}...")
                ranked_names = [line.strip().lstrip('- ').lstrip('1234567890. ') for line in response.splitlines() if line.strip()]
                logger.debug(f"Extracted {len(ranked_names)} ranked names from LLM response: {ranked_names[:5]}")
                
                model_map = {model['name']: model for model in models}
                logger.debug(f"Available model names in map: {list(model_map.keys())[:5]}")
                
                ranked_models = [model_map[name] for name in ranked_names if name in model_map]
                
                # Log which names didn't match
                unmatched = [name for name in ranked_names if name not in model_map]
                if unmatched:
                    logger.warning(f"LLM returned {len(unmatched)} names that don't match available models: {unmatched[:5]}")
                
                # If LLM ranking succeeded, return results
                if ranked_models:
                    logger.info(f"✅ LLM ranked {len(ranked_models)} models.")
                    return ranked_models[:15]
                
                logger.warning(f"LLM ranking returned no matching models. Falling back to relevance score sorting.")
        
        except Exception as e:
            logger.warning(f"LLM ranking failed: {e}. Falling back to relevance score sorting.")
        
        # Fallback: sort by relevance score
        return sorted(models, key=lambda x: x.get('relevance_score', 0), reverse=True)[:15]
    
    def download_model(self, model_info: Dict) -> Optional[Path]:
        """Download and extract a model, handling various archive types and structures."""
        model_name = model_info.get("name")
        download_url = model_info.get("download_url")
        if (not isinstance(model_name, str) or model_name in {"", ".", ".."} or
                Path(model_name).name != model_name or "\\" in model_name):
            raise ValueError("Invalid model name")
        if not self._allowed_download_url(download_url):
            raise ValueError("Model download provider is not allowed")
        destination_path = self.local_model_path / model_name
        if destination_path.is_symlink():
            raise ValueError("Model destination must not be a symbolic link")

        if destination_path.is_dir():
            try:
                self._validate_model_directory(destination_path)
            except (ValueError, OSError) as exc:
                logger.warning(f"Existing model '{model_name}' is invalid: {exc}")
            else:
                logger.info(f"Model '{model_name}' already exists. Skipping download.")
                return destination_path

        logger.info(f"Downloading '{model_name}' from {download_url}...")
        
        github_token = os.getenv("GITHUB_TOKEN")
        headers = ({'Authorization': f'token {github_token}'}
                   if github_token and urlparse(download_url).hostname in
                   {'github.com', 'api.github.com', 'codeload.github.com'} else {})
        response = self._make_request_with_retry(download_url, headers=headers, stream=True)
        if not response:
            logger.error(f"Failed to download '{model_name}' after multiple retries.")
            return None
        if not self._allowed_download_url(getattr(response, "url", download_url)):
            raise ValueError("Model download redirected to an unapproved provider")

        suffix = ".zip"
        if "zipball" in download_url:
            suffix = ".zip"
        elif "tarball" in download_url:
            suffix = ".tar.gz"

        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp_file:
            total_downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                total_downloaded += len(chunk)
                if total_downloaded > self.config.max_archive_bytes:
                    raise ValueError("Model archive exceeds size limit")
                tmp_file.write(chunk)
            tmp_file.seek(0)
            
            logger.info("Download complete. Extracting archive...")
            with tempfile.TemporaryDirectory(dir=self.local_model_path) as extract_dir_str:
                extract_dir = Path(extract_dir_str)
                try:
                    if suffix == ".zip":
                        with zipfile.ZipFile(tmp_file.name) as zf:
                            members = zf.infolist()
                            self._validate_archive_members(
                                [(m.filename, m.file_size,
                                  stat.S_ISLNK((m.external_attr >> 16) & 0xffff)) for m in members])
                            zf.extractall(extract_dir)
                    elif suffix == ".tar.gz":
                        with tarfile.open(tmp_file.name, "r:gz") as tf:
                            members = tf.getmembers()
                            self._validate_archive_members(
                                [(m.name, m.size, m.issym() or m.islnk() or
                                  not (m.isfile() or m.isdir())) for m in members])
                            for member in members:
                                tf.extract(member, extract_dir)
                    else:
                        logger.error(f"Unsupported archive type for {model_name}")
                        return None
                except (zipfile.BadZipFile, tarfile.TarError) as e:
                    logger.error(f"Failed to extract archive for {model_name}: {e}")
                    return None

                model_sdf_path = None
                target_path = model_info.get("path_in_repo")
                for potential_path in extract_dir.rglob("model.sdf"):
                    if target_path and not str(potential_path.parent).endswith(target_path):
                        continue
                    model_sdf_path = potential_path
                    break

                if model_sdf_path:
                    source_model_dir = model_sdf_path.parent
                    self._validate_model_directory(source_model_dir)
                    nonce = uuid.uuid4().hex
                    staged = self.local_model_path / f".{model_name}.{nonce}.staged"
                    shutil.move(source_model_dir, staged)
                    backup = self.local_model_path / f".{model_name}.{nonce}.backup"
                    try:
                        if destination_path.exists():
                            os.replace(destination_path, backup)
                        os.replace(staged, destination_path)
                    except Exception:
                        if backup.exists() and not destination_path.exists():
                            os.replace(backup, destination_path)
                        raise
                    finally:
                        if backup.exists() and destination_path.exists():
                            if backup.is_dir():
                                shutil.rmtree(backup)
                            else:
                                backup.unlink()
                    logger.info(f"✅ Successfully downloaded and verified model '{model_name}'.")
                    return destination_path
                else:
                    logger.error(f"Extraction failed: 'model.sdf' not found for '{model_name}'.")
                    return None

    def _allowed_download_url(self, url: str) -> bool:
        parsed = urlparse(url) if isinstance(url, str) else None
        return bool(parsed and parsed.scheme == "https" and
                    parsed.hostname in self.config.allowed_providers)

    @staticmethod
    def _validate_model_sdf(path: Path):
        root = ET.parse(path).getroot()
        model = root.find("model")
        if (root.tag != "sdf" or not root.get("version") or model is None or
                not model.get("name") or not any(
                    model.find(tag) is not None for tag in ("link", "include", "model"))):
            raise ValueError("model.sdf is not a valid model")

    def _validate_model_directory(self, directory: Path) -> None:
        try:
            self._validate_model_sdf(directory / "model.sdf")
            config_file = directory / "model.config"
            if not config_file.is_file():
                raise ValueError("Missing model.config")
            config_root = ET.parse(config_file).getroot()
            if (config_root.tag != "model" or not config_root.findtext("name") or
                    not config_root.findtext("sdf")):
                raise ValueError("Missing or invalid model.config")
            for sdf_entry in config_root.findall("sdf"):
                referenced = sdf_entry.text
                if not referenced:
                    raise ValueError("Empty SDF reference in model.config")
                relative = Path(referenced)
                if relative.is_absolute() or ".." in relative.parts or "\\" in referenced:
                    raise ValueError("Unsafe SDF reference in model.config")
                referenced_file = directory / relative
                if not referenced_file.is_file():
                    raise ValueError("Invalid SDF reference in model.config")
                self._validate_model_sdf(referenced_file)
            from gazebo_world_generator.src.models.visual_quality import inspect_model_visuals
            inspection = inspect_model_visuals(directory)
            if inspection.error:
                raise ValueError(f"Invalid model visual: {inspection.error}")
        except (ET.ParseError, OSError) as exc:
            raise ValueError("Invalid model XML") from exc

    def _validate_archive_members(self, members):
        if len(members) > 10_000:
            raise ValueError("Too many model archive members")
        total_size = 0
        for name, size, is_link in members:
            path = Path(name)
            if not name or "\\" in name or path.is_absolute() or ".." in path.parts or is_link:
                raise ValueError("Unsafe model archive member")
            total_size += size
            if total_size > self.config.max_archive_bytes:
                raise ValueError("Extracted model exceeds size limit")
