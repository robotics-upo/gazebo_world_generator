"""
Circuit Breaker Pattern Implementation

Prevents cascading failures by detecting failures and preventing calls to failing services.
Implements the three-state circuit breaker pattern: CLOSED -> OPEN -> HALF_OPEN -> CLOSED

Usage:
    breaker = CircuitBreaker(failure_threshold=5, timeout=60)

    @breaker.call
    def risky_operation():
        return api.call()
"""

import time
import logging
from enum import Enum
from typing import Callable, Optional, Any, TypeVar, Generic
from functools import wraps
from threading import Lock
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from gazebo_world_generator.src.exceptions import (
    CircuitBreakerOpenError,
    CircuitBreakerError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected due to open circuit
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: int = 0

    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_calls + self.failed_calls
        if total == 0:
            return 1.0
        return self.successful_calls / total

    def to_dict(self) -> dict:
        """Export stats as dictionary."""
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'rejected_calls': self.rejected_calls,
            'success_rate': self.success_rate(),
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_success': self.last_success_time.isoformat() if self.last_success_time else None,
            'state_changes': self.state_changes
        }


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures, requests are immediately rejected
    - HALF_OPEN: Testing recovery, limited requests allowed

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in HALF_OPEN to close circuit
        timeout: Seconds to wait before trying HALF_OPEN from OPEN
        expected_exception: Exception type that triggers failure count
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "default"
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures needed to open circuit
            success_threshold: Successes in HALF_OPEN to close circuit
            timeout: Seconds before attempting recovery
            expected_exception: Exception type that counts as failure
            name: Name for logging/monitoring
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")

        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = Lock()

        self.stats = CircuitBreakerStats()

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={failure_threshold}, "
            f"timeout={timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    def _change_state(self, new_state: CircuitState):
        """Change circuit state with logging."""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            self.stats.state_changes += 1
            logger.warning(
                f"Circuit breaker '{self.name}' state changed: "
                f"{old_state.value} -> {new_state.value}"
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.timeout

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If call fails
        """
        with self._lock:
            self.stats.total_calls += 1

            # Check if we should attempt recovery
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(
                        f"Circuit breaker '{self.name}' attempting recovery "
                        f"(HALF_OPEN)"
                    )
                    self._change_state(CircuitState.HALF_OPEN)
                    self._success_count = 0
                else:
                    # Circuit still open, reject call
                    self.stats.rejected_calls += 1
                    time_until_retry = self.timeout - (time.time() - self._last_failure_time)
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Retry in {time_until_retry:.1f}s",
                        details={
                            'circuit_name': self.name,
                            'state': self._state.value,
                            'failure_count': self._failure_count,
                            'time_until_retry': time_until_retry
                        }
                    )

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.stats.successful_calls += 1
            self.stats.last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit breaker '{self.name}' success in HALF_OPEN: "
                    f"{self._success_count}/{self.success_threshold}"
                )

                if self._success_count >= self.success_threshold:
                    logger.info(
                        f"Circuit breaker '{self.name}' recovered, closing circuit"
                    )
                    self._change_state(CircuitState.CLOSED)
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in CLOSED state
                self._failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.stats.failed_calls += 1
            self.stats.last_failure_time = datetime.now()
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self.name}' failure: "
                f"{self._failure_count}/{self.failure_threshold}"
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN reopens the circuit
                logger.warning(
                    f"Circuit breaker '{self.name}' failed in HALF_OPEN, "
                    f"reopening circuit"
                )
                self._change_state(CircuitState.OPEN)
            elif self._failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker '{self.name}' threshold exceeded, "
                    f"opening circuit"
                )
                self._change_state(CircuitState.OPEN)

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"Circuit breaker '{self.name}' manually reset")
            self._change_state(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self.stats

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for wrapping functions with circuit breaker.

        Usage:
            @circuit_breaker
            def my_function():
                return risky_call()
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized management and monitoring of all circuit breakers.
    """

    def __init__(self):
        """Initialize circuit breaker registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = Lock()

    def register(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ) -> CircuitBreaker:
        """
        Register a new circuit breaker.

        Args:
            name: Unique name for the circuit breaker
            failure_threshold: Failures before opening
            success_threshold: Successes to close from HALF_OPEN
            timeout: Recovery timeout in seconds
            expected_exception: Exception type for failures

        Returns:
            Registered circuit breaker
        """
        with self._lock:
            if name in self._breakers:
                logger.warning(f"Circuit breaker '{name}' already exists, returning existing")
                return self._breakers[name]

            breaker = CircuitBreaker(
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=timeout,
                expected_exception=expected_exception,
                name=name
            )
            self._breakers[name] = breaker
            return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats().to_dict()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("All circuit breakers reset")


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 60.0,
    expected_exception: type = Exception
) -> CircuitBreaker:
    """
    Get or create a circuit breaker from global registry.

    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        success_threshold: Successes to close
        timeout: Recovery timeout
        expected_exception: Exception type for failures

    Returns:
        Circuit breaker instance
    """
    return _registry.register(
        name=name,
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        expected_exception=expected_exception
    )


def get_all_circuit_breaker_stats() -> dict[str, dict]:
    """Get statistics for all registered circuit breakers."""
    return _registry.get_all_stats()
