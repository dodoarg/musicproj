from typing import Any

from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import HTMLResponse
from app.config import settings

from http.server import HTTPServer

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}.openapi.json"
)

root_router = APIRouter()

@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response"""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API<\h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)

app.include_router(root_router)

if __name__ == "__main__":
    # use this for debugging purposes only

    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)