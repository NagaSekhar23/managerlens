"""Provider-agnostic error types for LLM calls, mapped to HTTP responses in routers."""


class LLMError(Exception):
    """Base class for all LLM-call failures."""


class MissingAPIKeyError(LLMError):
    """Raised when no API key is configured for the LLM provider."""


class LLMRequestError(LLMError):
    """Raised when the call to the LLM provider fails (network, auth, rate limit, etc.)."""


class InvalidLLMOutputError(LLMError):
    """Raised when the LLM's response cannot be parsed into the expected structured schema."""


class KnowledgeRetrievalError(Exception):
    """Raised when the knowledge-base query itself fails (DB connection, query error, etc.).

    Distinct from LLMError: this is a storage failure, not a model-provider failure. Callers
    may choose to degrade gracefully (skip retrieval, proceed without evidence) rather than
    fail the whole request, since RAG evidence is a supporting input, not a hard dependency.
    """


class EmployeeNotFoundError(Exception):
    """Raised when a company-data lookup is requested for an employee_id that doesn't exist
    in the mock company dataset. Callers should handle this gracefully rather than crash the
    analysis request."""
