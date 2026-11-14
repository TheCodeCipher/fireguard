from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fireguard.api.routes import events, firms, health, live, replays
from fireguard.core.config import get_settings
from fireguard.core.errors import APIError

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "FireGuard backend API. FIRMS routes serve satellite observations; "
        "historical forecasts are retrieved as precomputed out-of-fold artifacts."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Replay-Session-ID"],
)


@app.exception_handler(APIError)
async def api_error_handler(_request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "One or more request parameters are invalid.",
                "details": jsonable_encoder(exc.errors()),
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_request: Request, exc: StarletteHTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": f"HTTP_{exc.status_code}", "message": message}},
    )


app.include_router(health.router, prefix=settings.api_v1_prefix)
app.include_router(firms.router, prefix=settings.api_v1_prefix)
app.include_router(events.router, prefix=settings.api_v1_prefix)
app.include_router(replays.router, prefix=settings.api_v1_prefix)
app.include_router(replays.artifacts_router, prefix=settings.api_v1_prefix)
app.include_router(live.router, prefix=settings.api_v1_prefix)


@app.get("/", tags=["Root"])
async def root() -> dict:
    return {
        "message": "Welcome to the FireGuard API",
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
        "firms_observations_mock": f"{settings.api_v1_prefix}/firms/observations",
        "fires": f"{settings.api_v1_prefix}/fires",
        "events": f"{settings.api_v1_prefix}/wsts/events",
    }
