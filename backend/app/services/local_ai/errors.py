"""Safe, stable errors for local AI operations."""

from __future__ import annotations


class LocalAIError(RuntimeError):
    """Base error for local AI failures safe to expose by code only."""

    code = "local_ai_error"
    retryable = False


class LocalPolicyError(LocalAIError):
    """Raised when requested routing crosses a local-policy boundary."""

    code = "local_policy_error"


class LocalWorkerError(LocalAIError):
    """Raised for a worker process failure."""

    code = "local_worker_error"


class LocalWorkerTimeout(LocalWorkerError):
    """Raised when a local worker exceeds its execution deadline."""

    code = "local_worker_timeout"
    retryable = True


class LocalValidationError(LocalAIError):
    """Raised when local data or artifacts fail validation."""

    code = "local_validation_error"
