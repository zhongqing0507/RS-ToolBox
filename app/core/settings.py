# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : settings.py
# Time       ：2024/9/10 11:05
# Author     ：author zhongq
# Description：
"""
import os
import configparser
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings


def load_config_file_env(config_file: str):
    if os.path.exists(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        # env = os.getenv('ENV')
        # if 'global' in config:
        #     env = config.get('global', 'environment', fallback='dev')
        #     os.environ['ENV'] = env
        env = os.environ.get("ENVIRONMENT", config.get("global", "environment")).lower()
        os.environ['ENV'] = env
        if env in config:
            for key, value in config[env].items():
                os.environ[key.upper()] = value

    if os.getenv('ENV') is None:
        os.environ['ENV'] = 'dev'


class Settings(BaseSettings):
    DEBUG: bool = False

    APP_NAME: str = 'RS-Toolbox'
    APP_VERSION: str = '0.0.1'
    API_V1_STR: str = '/api/v1'
    CACHE_CONTROL: str = "public, max-age=2592000"

    ENABLE_LOWER_CASE_QUERY_PARAM: bool = True
    ENABLE_PROMETHEUS_INSTRUMENT: bool = False
    ENABLE_TRACING: bool = False

    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    # 用于记录捕获的错误信息
    # ERROR_MESSAGES = dict()

    model_path: str = os.getenv('model_path')
    share_path: str = os.getenv("share_path")

    @field_validator("BACKEND_CORS_ORIGINS")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)



load_config_file_env('config.ini')
app_settings = Settings()
