# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : api_base.py
# Time       ：2024/9/10 11:08
# Author     ：author zhongq
# Description：
"""
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse
from app.core import app_settings
from fastapi.responses import JSONResponse
api_base_router = APIRouter()


@api_base_router.get("/")
async def root(response: Response):
    return f'hello {app_settings.APP_NAME}'


@api_base_router.get("/health")
async def health_check(response: Response) -> Response:
    """Health check."""
    # 如果有捕获的错误信息，服务被视为不健康
    if "RuntimeError" in app_settings.ERROR_MESSAGES:
        return JSONResponse(content={"status": "unhealthy", "errors": app_settings.ERROR_MESSAGES["RuntimeError"]}, status_code=500)
    else:
        # 没有错误信息，服务被视为健康
        return JSONResponse(content={"status": "healthy"}, status_code=200)


@api_base_router.get("/version")
async def version(response: Response):
    """TiTiler Landing page"""
    return app_settings.APP_VERSION
