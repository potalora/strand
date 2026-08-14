"""Safe, stable errors for local AI operations."""

from __future__ import annotations


LOCAL_WORKER_FAILURE_CATEGORIES = frozenset(
    {
        "fragment_conflict",
        "invalid_structured_output",
        "output_limit",
        "stream_contract",
        "work_limit",
        "work_token_limit",
        "work_attempt_limit",
        "work_split_limit",
        "fragment_depth_limit",
    }
)


class LocalAIError(RuntimeError):
    """Base error for local AI failures safe to expose by code only."""

    code = "local_ai_error"
    retryable = False


class LocalPolicyError(LocalAIError):
    """Raised when requested routing crosses a local-policy boundary."""

    code = "local_policy_error"


class RuntimeIdentityRequiredError(LocalPolicyError):
    """A legacy strict-local snapshot cannot be admitted after v2 rollout."""

    code = "runtime_identity_required"


class LocalWorkerError(LocalAIError):
    """Raised for a worker process failure."""

    code = "local_worker_error"

    def __init__(self, message: str, *, category: str | None = None) -> None:
        if category is not None and category not in LOCAL_WORKER_FAILURE_CATEGORIES:
            raise ValueError("Local worker failure category is invalid.")
        super().__init__(message)
        self.category = category


class LocalInputLimitError(LocalWorkerError):
    """Raised when valid local input cannot fit the locked model limits."""

    code = "local_input_limit_exceeded"


class LocalWorkerTimeout(LocalWorkerError):
    """Raised when a local worker exceeds its execution deadline."""

    code = "local_worker_timeout"
    retryable = True


class LocalValidationError(LocalAIError):
    """Raised when local data or artifacts fail validation."""

    code = "local_validation_error"
