# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : __init__.py.py
# Time       ：2024/9/10 11:08
# Author     ：author zhongq
# Description：
"""
from fastapi import APIRouter
from app.api.api_base import api_base_router
from app.api.api_raster import api_raster_router

api_router = APIRouter()
api_router.include_router(api_raster_router, prefix="/raster", tags=["raster toolbox"])


__all__ = ['api_router', 'api_base_router']
