"""Script to run the FastAPI backend server."""

import uvicorn
from backend.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=settings.fastapi_reload,
        log_level=settings.log_level.lower()
    )
