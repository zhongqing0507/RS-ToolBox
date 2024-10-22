# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : response.py
# Time       ：2024/9/10 11:05
# Author     ：author zhongq
# Description：
"""
from fastapi import status
from fastapi.responses import JSONResponse
from typing import Union, Any, Optional, Dict
from fastapi.encoders import jsonable_encoder


# 定义响应统一结构体
class AppJSONResponse(JSONResponse):
    def __init__(
        self,
        content: Any,
        status_code: int = status.HTTP_200_OK,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(content, status_code, headers)


def resp_json(
    code: int = 0,
    message: Optional[Any] = None,
    data: Union[list, dict, str, None] = None,
):
    return {
        'code': code,
        'message': message,
        'data': data,
    }

def resp_ok(data: Union[list, dict, str]):
    """
    200系列的响应结构体
    *：代表调用方法时必须传参数
    Union：代表传入的参数可以是多种类型
    """
    return AppJSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder({
            'code': 0,
            'message': "success",
            'data': data,
        })
    )


def resp_bad_request(code: int = 1, data: str = "error", message: Optional[Any] = None):
    """
    400系列的响应结构体
    *：代表调用方法时必须传参数
    """
    return AppJSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({
            'code': 1,
            'message': message,
            'data': data,
        })
    )
