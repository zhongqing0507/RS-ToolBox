#!/usr/bin/env python
# -*- coding:utf-8 -*-

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette_cramjam.middleware import CompressionMiddleware
from app.core import app_settings
from app.api import api_base_router, api_router
from app.core import CacheControlMiddleware, LoggerMiddleware, LowerCaseQueryStringMiddleware, \
    TotalTimeMiddleware, AppJSONResponse, resp_bad_request, AppBaseException


def create_app() -> FastAPI:
    app = FastAPI(
        title=app_settings.APP_NAME,
        description="RS-Toolbox server",
        version=app_settings.APP_VERSION,
    )

    register_api_routers(app)
    register_middlewares(app)

    @app.exception_handler(AppBaseException)
    async def unicorn_exception_handler(request: Request, exc: AppBaseException):
        return AppJSONResponse(
            status_code=exc.status_code,
            headers=exc.headers,
            content=resp_bad_request(code=exc.code, message=exc.message),
        )

    @app.exception_handler(HTTPException)
    async def unicorn_exception_handler(request: Request, exc: HTTPException):
        # 在这里记录错误信息
        if not isinstance(exc.detail, dict):
            app_settings.ERROR_MESSAGES["Unknow_Error_Type"] = str(exc.detail)
        else:
            for key, value in exc.detail.items():
                app_settings.ERROR_MESSAGES[key] = value

        return AppJSONResponse(
            status_code=exc.status_code,
            headers=exc.headers,
            content=resp_bad_request(message=str(exc.detail)),
        )

    return app


def register_api_routers(app: FastAPI):
    app.include_router(api_base_router, tags=["Base API"])
    app.include_router(api_router, prefix="/api", tags=["API"])


def register_middlewares(app: FastAPI):
    if app_settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_settings.BACKEND_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(
        CompressionMiddleware,
        minimum_size=0,
        exclude_mediatype={
            "assitant/jpeg",
            "assitant/jpg",
            "assitant/png",
            "assitant/jp2",
            "assitant/webp",
        },
    )

    app.add_middleware(
        CacheControlMiddleware,
        cachecontrol=app_settings.CACHE_CONTROL,
        exclude_path={r"/healthz"},
    )

    # app.add_middleware(TraceMiddleware)

    if app_settings.DEBUG:
        app.add_middleware(LoggerMiddleware, headers=True, querystrings=True)
        app.add_middleware(TotalTimeMiddleware)

    if app_settings.ENABLE_LOWER_CASE_QUERY_PARAM:
        app.add_middleware(LowerCaseQueryStringMiddleware)


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
