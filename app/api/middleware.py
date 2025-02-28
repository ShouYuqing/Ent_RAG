"""
Middleware for the API layer.
Includes API key authentication middleware.
"""

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import config


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    async def dispatch(self, request: Request, call_next):
        """
        Check for valid API key in the request headers.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response from the next middleware or route handler
            
        Raises:
            HTTPException: If the API key is missing or invalid
        """
        # Skip authentication for documentation endpoints
        if request.url.path in ["/docs", "/redoc", "/openapi.json", "/health", "/"]:
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key")
        
        # Check if API key is provided and valid
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is missing",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        if api_key != config.api.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        # Continue processing the request
        return await call_next(request) 