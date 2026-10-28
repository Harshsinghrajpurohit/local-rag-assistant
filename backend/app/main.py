"""
FastAPI application initialization and global exception handlers.
Sets up:
- FastAPI app with title and version
- Global exception handler for APIException subclasses
- Request ID generation for correlation
- Health check endpoint
"""

from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .exceptions import APIException, FileTooLargeError, InvalidMimeTypeError, \
    PdfCorruptedError, DocumentNotFoundError, OllamaUnavailableError, \
    DeleteFailedError, DeletePartialError, InternalServerError, InvalidRequestError
from .config import Config
from .models import ErrorResponse, generate_request_id

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="RAG-Q&A Document Assistant",
        version="5.0.0",
        description="Retrieval-Augmented Generation Q&A system with hybrid search "
                    "(BM25 + dense) and timezone-aware UTC timestamps"
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint for deployment verification."""
        return {"status": "healthy", "service": "RAG-Q&A API"}

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        """
        Global exception handler for all APIException subclasses.

        Ensures consistent error responses with:
        - Structured error type
        - User-friendly detail message
        - request_id for log correlation
        - ISO 8601 UTC timestamp
        - Appropriate HTTP status code
        """
        error_response = ErrorResponse(
            error=exc.error_type,
            detail=exc.detail,
            request_id=exc.request_id,
            timestamp=generate_request_id()  # Use helper or replace with actual timestamp
        )
        # Actually let's use the proper timestamp
        error_response.timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump()
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """
        Fallback handler for unexpected exceptions.

        Returns a generic 500 error without exposing internal details.
        """
        from .models import generate_request_id
        error_response = ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            detail="An unexpected error occurred",
            request_id=generate_request_id(),
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )

    # Register routers
    from .api.chat import router as chat_router
    from .api.documents import router as documents_router

    app.include_router(chat_router)
    app.include_router(documents_router)

    return app


# Create app instance for direct running
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)