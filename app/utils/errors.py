class NexusException(Exception):
    code = "INTERNAL_ERROR"
    status_code = 500
    message = "Internal server error"

    def __init__(self, message: str | None = None):
        super().__init__(message or self.message)
        self.message = message or self.message


class ValidationException(NexusException):
    code, status_code, message = "VALIDATION_ERROR", 422, "Invalid request"


class NotFoundException(NexusException):
    code, status_code, message = "NOT_FOUND", 404, "Resource not found"


class LLMException(NexusException):
    code, status_code, message = "LLM_FAILED", 502, "Language model request failed"


class RetrievalException(NexusException):
    code, status_code, message = "RETRIEVAL_FAILED", 502, "Knowledge retrieval failed"


class VectorDatabaseException(NexusException):
    code, status_code, message = "VECTOR_DB_FAILED", 502, "Vector database request failed"


class DatabaseException(NexusException):
    code, status_code, message = "DATABASE_FAILED", 503, "Database request failed"


class DocumentProcessingException(NexusException):
    code, status_code, message = "DOCUMENT_PROCESSING_FAILED", 422, "Document processing failed"


class ToolExecutionException(NexusException):
    code, status_code, message = "TOOL_EXECUTION_FAILED", 502, "Tool execution failed"