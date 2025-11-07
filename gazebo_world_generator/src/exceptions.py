"""
Custom exceptions for Gazebo World Generator.

Only contains exceptions that are actually used in production code.
"""


# =============================================================================
# Configuration Errors
# =============================================================================

class InvalidConfigError(Exception):
    """Configuration file is invalid or malformed."""
    pass


class MissingConfigError(Exception):
    """Required configuration parameter is missing."""
    pass


class ConfigValidationError(Exception):
    """Configuration validation failed."""
    pass


# =============================================================================
# Circuit Breaker Errors
# =============================================================================

class CircuitBreakerError(Exception):
    """Base class for circuit breaker errors."""
    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Circuit breaker is open, rejecting requests."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# =============================================================================
# LLM Errors (used only in tests)
# =============================================================================

class LLMConnectionError(Exception):
    """LLM server is unreachable or connection failed."""
    pass


class LLMTimeoutError(Exception):
    """LLM request timed out."""
    pass


class JSONParseError(Exception):
    """Failed to parse JSON from LLM response."""
    pass


# =============================================================================
# Schema Validation Errors  
# =============================================================================

class SchemaValidationError(Exception):
    """LLM response failed JSON schema validation."""
    pass


# =============================================================================
# Model Errors (used only in tests)
# =============================================================================

class ModelNotFoundError(Exception):
    """Required model could not be found."""
    pass


class CollisionError(Exception):
    """Unable to place objects without collisions."""
    pass


class InsufficientSpaceError(Exception):
    """Room is too small to fit all requested objects."""
    pass


# =============================================================================
# Resource Limit Errors
# =============================================================================

class ResourceLimitExceededError(Exception):
    """Request exceeds system resource limits."""

    def __init__(self, resource_type: str, requested: int, limit: int):
        message = f"{resource_type} limit exceeded: requested {requested}, limit is {limit}"
        super().__init__(message)
        self.resource_type = resource_type
        self.requested = requested
        self.limit = limit
