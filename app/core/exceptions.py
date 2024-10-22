# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : exceptions.py
# Time       ：2024/9/10 11:04
# Author     ：author zhongq
# Description：
"""
import http
from typing import Any, Optional
from fastapi import status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


class AppBaseException(Exception):
    def __init__(
            self,
            status_code: int,
            code: int = 1,
            headers: Optional[dict] = None,
            message: Optional[Any] = None,
    ) -> None:
        if message is None:
            message = http.HTTPStatus(status_code).phrase
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers


class HTTPRequestParameterError(AppBaseException):
    def __init__(
            self,
            message: Optional[Any] = None,
    ) -> None:
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, message=message)


class HTTPBadRequestError(AppBaseException):
    def __init__(
            self,
            code: int = 1,
            message: Optional[Any] = None,
    ) -> None:
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, code=code, message=message)
