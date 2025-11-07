"""
Prompt template management system with Jinja2.

Features:
- Template versioning (v1, v2, experimental)
- A/B testing support
- Performance tracking
- Multi-language support potential
- Easy customization without code changes
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

logger = logging.getLogger(__name__)


@dataclass
class PromptMetrics:
    """Metrics for tracking prompt performance."""

    template_name: str
    version: str
    successes: int = 0
    failures: int = 0
    total_tokens: int = 0
    avg_response_time: float = 0.0
    last_used: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successes + self.failures
        return (self.successes / total * 100) if total > 0 else 0.0

    @property
    def usage_count(self) -> int:
        """Total usage count."""
        return self.successes + self.failures

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'template_name': self.template_name,
            'version': self.version,
            'successes': self.successes,
            'failures': self.failures,
            'total_tokens': self.total_tokens,
            'avg_response_time': self.avg_response_time,
            'success_rate': self.success_rate,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }


@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata."""

    name: str
    version: str
    template: Template
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def render(self, **kwargs) -> str:
        """
        Render the template with given context.

        Args:
            **kwargs: Template variables

        Returns:
            Rendered prompt string
        """
        try:
            return self.template.render(**kwargs)
        except Exception as e:
            logger.error(f"Failed to render template {self.name}: {e}")
            raise


class PromptManager:
    """
    Manages prompt templates with versioning and performance tracking.

    Features:
    - Load templates from directory structure
    - Version management (v1, v2, experimental)
    - A/B testing support
    - Performance metrics tracking
    - Automatic fallback to previous versions
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        default_version: str = 'v1',
        enable_metrics: bool = True,
        metrics_file: Optional[Path] = None
    ):
        """
        Initialize prompt manager.

        Args:
            template_dir: Directory containing template versions
            default_version: Default template version to use
            enable_metrics: Enable performance tracking
            metrics_file: File to persist metrics
        """
        # Set template directory
        if template_dir is None:
            # Try ROS2 share directory first (for installed packages)
            try:
                from ament_index_python.packages import get_package_share_directory
                share_dir = Path(get_package_share_directory('gazebo_world_generator'))
                template_dir = share_dir / 'prompts'
                if not template_dir.exists():
                    raise FileNotFoundError("Share directory prompts not found")
            except (ImportError, FileNotFoundError, Exception) as e:
                # Fallback to source directory (for development)
                logger.debug(f"ROS2 share directory not available ({e}), using source directory")
                pkg_dir = Path(__file__).parent.parent.parent.parent
                template_dir = pkg_dir / 'prompts'
        else:
            # If template_dir is provided, resolve it relative to package directory if it's a relative path
            template_dir = Path(template_dir)
            if not template_dir.is_absolute():
                # Try ROS2 share directory first
                try:
                    from ament_index_python.packages import get_package_share_directory
                    share_dir = Path(get_package_share_directory('gazebo_world_generator'))
                    template_dir = share_dir / template_dir
                    if not template_dir.exists():
                        raise FileNotFoundError("Share directory template_dir not found")
                except (ImportError, FileNotFoundError, Exception):
                    # Fallback to source directory
                    pkg_dir = Path(__file__).parent.parent.parent.parent
                    template_dir = pkg_dir / template_dir

        self.template_dir = template_dir
        self.default_version = default_version
        self.enable_metrics = enable_metrics

        # Set up metrics file
        if metrics_file is None:
            metrics_file = self.template_dir / '.metrics.json'
        self.metrics_file = Path(metrics_file)

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )

        # Add custom filters
        self._add_custom_filters()

        # Load metrics
        self.metrics: Dict[str, PromptMetrics] = {}
        if self.enable_metrics:
            self._load_metrics()

        # Cache loaded templates
        self._template_cache: Dict[str, PromptTemplate] = {}

        logger.info(f"PromptManager initialized with template_dir={self.template_dir}, "
                   f"default_version={self.default_version}")

    def _add_custom_filters(self):
        """Add custom Jinja2 filters."""

        def format_list(items: List[str], separator: str = ', ', last_separator: str = ' and ') -> str:
            """Format a list with proper separators."""
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            if len(items) == 2:
                return f"{items[0]}{last_separator}{items[1]}"
            return separator.join(items[:-1]) + last_separator + items[-1]

        def truncate_smart(text: str, length: int = 100) -> str:
            """Truncate text at word boundary."""
            if len(text) <= length:
                return text
            truncated = text[:length].rsplit(' ', 1)[0]
            return truncated + '...'

        self.env.filters['format_list'] = format_list
        self.env.filters['truncate_smart'] = truncate_smart

    def get_template(
        self,
        template_name: str,
        version: Optional[str] = None,
        fallback_to_default: bool = True
    ) -> PromptTemplate:
        """
        Get a prompt template.

        Args:
            template_name: Name of the template (without .j2 extension)
            version: Template version (defaults to self.default_version)
            fallback_to_default: Fallback to default version if not found

        Returns:
            PromptTemplate instance

        Raises:
            TemplateNotFound: If template doesn't exist
        """
        version = version or self.default_version
        cache_key = f"{version}/{template_name}"

        # Check cache
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        # Try to load template
        template_path = f"{version}/{template_name}.j2"
        try:
            jinja_template = self.env.get_template(template_path)

            # Create PromptTemplate
            prompt_template = PromptTemplate(
                name=template_name,
                version=version,
                template=jinja_template,
                description=f"Template: {template_name} (version {version})"
            )

            # Cache it
            self._template_cache[cache_key] = prompt_template
            logger.debug(f"Loaded template: {template_path}")

            return prompt_template

        except TemplateNotFound:
            if fallback_to_default and version != self.default_version:
                logger.warning(f"Template {template_path} not found, falling back to {self.default_version}")
                return self.get_template(template_name, self.default_version, fallback_to_default=False)
            raise TemplateNotFound(f"Template not found: {template_path}")

    def render(
        self,
        template_name: str,
        version: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Render a prompt template.

        Args:
            template_name: Name of the template
            version: Template version
            **kwargs: Template variables

        Returns:
            Rendered prompt string
        """
        template = self.get_template(template_name, version)
        return template.render(**kwargs)

    def log_success(
        self,
        template_name: str,
        version: Optional[str] = None,
        tokens_used: int = 0,
        response_time: float = 0.0
    ):
        """
        Log successful template usage.

        Args:
            template_name: Template name
            version: Template version
            tokens_used: Number of tokens used
            response_time: Response time in seconds
        """
        if not self.enable_metrics:
            return

        version = version or self.default_version
        key = f"{version}/{template_name}"

        if key not in self.metrics:
            self.metrics[key] = PromptMetrics(
                template_name=template_name,
                version=version
            )

        metrics = self.metrics[key]
        metrics.successes += 1
        metrics.total_tokens += tokens_used
        metrics.last_used = datetime.now()

        # Update average response time
        total_responses = metrics.successes + metrics.failures
        if total_responses > 0:
            metrics.avg_response_time = (
                (metrics.avg_response_time * (total_responses - 1) + response_time) / total_responses
            )

        # Save metrics periodically (every 10 uses)
        if total_responses % 10 == 0:
            self._save_metrics()

    def log_failure(
        self,
        template_name: str,
        version: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Log failed template usage.

        Args:
            template_name: Template name
            version: Template version
            error: Optional error message
        """
        if not self.enable_metrics:
            return

        version = version or self.default_version
        key = f"{version}/{template_name}"

        if key not in self.metrics:
            self.metrics[key] = PromptMetrics(
                template_name=template_name,
                version=version
            )

        self.metrics[key].failures += 1
        self.metrics[key].last_used = datetime.now()

        if error:
            logger.warning(f"Template {key} failed: {error}")

    def get_metrics(self, template_name: Optional[str] = None) -> Dict[str, Dict]:
        """
        Get performance metrics.

        Args:
            template_name: Optional filter by template name

        Returns:
            Dictionary of metrics
        """
        if template_name:
            return {
                k: v.to_dict()
                for k, v in self.metrics.items()
                if v.template_name == template_name
            }
        return {k: v.to_dict() for k, v in self.metrics.items()}

    def get_best_version(self, template_name: str) -> str:
        """
        Get the best performing version of a template.

        Args:
            template_name: Template name

        Returns:
            Best version string
        """
        relevant_metrics = [
            (key, metrics)
            for key, metrics in self.metrics.items()
            if metrics.template_name == template_name
        ]

        if not relevant_metrics:
            return self.default_version

        # Sort by success rate, then by usage count
        best = max(
            relevant_metrics,
            key=lambda x: (x[1].success_rate, x[1].usage_count)
        )

        return best[1].version

    def list_templates(self, version: Optional[str] = None) -> List[str]:
        """
        List available templates.

        Args:
            version: Optional version filter

        Returns:
            List of template names
        """
        version = version or self.default_version
        version_dir = self.template_dir / version

        if not version_dir.exists():
            logger.warning(f"Version directory not found: {version_dir}")
            return []

        templates = []
        for template_file in version_dir.glob('*.j2'):
            templates.append(template_file.stem)

        return sorted(templates)

    def list_versions(self) -> List[str]:
        """
        List available template versions.

        Returns:
            List of version strings
        """
        versions = []
        for version_dir in self.template_dir.iterdir():
            if version_dir.is_dir() and not version_dir.name.startswith('.'):
                versions.append(version_dir.name)

        return sorted(versions)

    def _load_metrics(self):
        """Load metrics from file."""
        if not self.metrics_file.exists():
            return

        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)

            for key, metrics_dict in data.items():
                self.metrics[key] = PromptMetrics(
                    template_name=metrics_dict['template_name'],
                    version=metrics_dict['version'],
                    successes=metrics_dict.get('successes', 0),
                    failures=metrics_dict.get('failures', 0),
                    total_tokens=metrics_dict.get('total_tokens', 0),
                    avg_response_time=metrics_dict.get('avg_response_time', 0.0),
                    last_used=datetime.fromisoformat(metrics_dict['last_used'])
                    if metrics_dict.get('last_used') else None
                )

            logger.info(f"Loaded metrics for {len(self.metrics)} templates")

        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")

    def _save_metrics(self):
        """Save metrics to file."""
        try:
            # Ensure directory exists
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

            # Save metrics
            data = {k: v.to_dict() for k, v in self.metrics.items()}

            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved metrics to {self.metrics_file}")

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def __del__(self):
        """Save metrics on destruction."""
        if self.enable_metrics and self.metrics:
            self._save_metrics()


# Create global instance
_default_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global PromptManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PromptManager()
    return _default_manager


def set_prompt_manager(manager: PromptManager):
    """Set the global PromptManager instance."""
    global _default_manager
    _default_manager = manager
