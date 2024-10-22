# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : raster.py
# Time       ：2024/9/10 14:44
# Author     ：author zhongq
# Description：
"""
from typing import Optional,Union, Tuple
from pydantic import BaseModel, Field


class SplitGridsModel(BaseModel):
    image: str = Field(..., description="split raster image file path")
    window_size: Optional[Tuple[int, int]] = Field(default=(512, 512), description="window size")
    overlap_ratio: Optional[Tuple[float, float]] = Field(default=(0.2, 0.2), description="stride size")
    output_path: Union[str, None] = Field(None, description="save path")


class MergeGridsModel(BaseModel):
    meta_file: str = Field(..., description="merge raster image file path")


class VectorModel(BaseModel):
    meta_file: str = Field(..., description="merge raster image file path")
    specific_label: Union[str, None] = Field(None, description="specific label")