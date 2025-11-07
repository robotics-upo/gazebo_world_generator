"""
Simple JSON validation for LLM responses.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation with detailed error information."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parsed_data: Optional[Any] = None

    def __bool__(self) -> bool:
        """Allow boolean evaluation."""
        return self.is_valid

    @property
    def error_summary(self) -> str:
        """Get a summary of errors."""
        if not self.errors:
            return "No errors"
        return f"{len(self.errors)} error(s): " + "; ".join(self.errors[:3])


class JSONValidator:
    """Validates that response is valid JSON with required keys."""

    def __init__(self, required_keys: Optional[List[str]] = None):
        """
        Initialize JSON validator.

        Args:
            required_keys: Optional list of required top-level keys
        """
        self.required_keys = required_keys or []

    def validate(self, response: str) -> ValidationResult:
        """
        Validate JSON format and required keys.
        
        Args:
            response: JSON string to validate
            
        Returns:
            ValidationResult with validation status and errors
        """
        errors = []
        warnings = []
        parsed_data = None

        # Try parsing JSON
        try:
            parsed_data = json.loads(response)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                parsed_data=None
            )

        # Check required keys
        if self.required_keys and isinstance(parsed_data, dict):
            missing_keys = [key for key in self.required_keys if key not in parsed_data]
            if missing_keys:
                errors.append(f"Missing required keys: {', '.join(missing_keys)}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            parsed_data=parsed_data
        )

    def get_validation_hints(self, errors: List[str]) -> str:
        """
        Get hints for fixing validation errors.

        Args:
            errors: List of error messages

        Returns:
            String with hints for correction
        """
        hints = []

        for error in errors:
            if "Invalid JSON" in error:
                hints.append("- Ensure the response is valid JSON")
                hints.append("- Check for missing commas, brackets, or quotes")
                hints.append("- Avoid trailing commas")
            elif "Missing required keys" in error:
                hints.append(f"- Include all required keys: {', '.join(self.required_keys)}")
                hints.append("- Verify the JSON structure matches the expected schema")

        if not hints:
            hints.append("- Review the JSON structure and fix any syntax errors")

        return "\n".join(hints)
