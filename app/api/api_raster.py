# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : api_raster.py
# Time       ：2024/9/10 11:18
# Author     ：author zhongq
# Description：
"""

from fastapi import APIRouter
from app.rstoolbox import RSGrids, raster_to_vector
from app.schemas import SplitGridsModel, MergeGridsModel, VectorModel
from app.core import resp_ok, resp_bad_request
from app.utils import LOG
api_raster_router = APIRouter()


@api_raster_router.post("/split_grids")
async def split_grids(sg_model: SplitGridsModel):
    try:
        LOG.info(f"=====Process Split Grids {sg_model.image}=======")
        rs_grids = RSGrids.from_image(sg_model.image, sg_model.output_path)
        meta_file = rs_grids.split_grids(
            window_size=sg_model.window_size,
            overlap_ratio=sg_model.overlap_ratio)
        LOG.info(f"=====Process Split Grids Success=======")
        return resp_ok(data=meta_file)
    except Exception as err:
        LOG.error(f"=====Process Split Grids Failed=======\n{err}")
        return resp_bad_request(message=str(err))


@api_raster_router.post("/merge_grids")
async def merge_grids(mg_model: MergeGridsModel):
    try:
        LOG.info(f"=====Process Merge Grids {mg_model.meta_file}=======")
        meta_file = RSGrids.merge_grids(mg_model.meta_file)
        LOG.info(f"=====Process Merge Grids Success=======")
        return resp_ok(data=meta_file)
    except Exception as err:
        LOG.error(f"=====Process Merge Grids Failed=======\n{err}")
        return resp_bad_request(message=str(err))


@api_raster_router.post("/raster_to_vector")
async def raster2vector(v_model: VectorModel):
    try:
        LOG.info(f"=====Process Raster to Vector {v_model.meta_file}=======")
        meta_file = await raster_to_vector(v_model.meta_file, v_model.specific_label)
        LOG.info(f"=====Process Raster to Vector Success=======")
        return resp_ok(data=meta_file)
    except Exception as err:
        LOG.error(f"=====Process Raster to Vector Failed=======\n{err}")
        return resp_bad_request(message=str(err))

@api_raster_router.post("/det_visualize")
async def det_visualize(mg_model: MergeGridsModel):
    try:
        LOG.info(f"=====Process visualize {mg_model.meta_file}=======")
        meta_file = RSGrids.draw_det_instances(mg_model.meta_file)
        LOG.info(f"=====Process visualize Success=======")
        return resp_ok(data=meta_file)
    except Exception as err:
        LOG.error(f"=====Process visualize Failed=======\n{err}")
        return resp_bad_request(message=str(err))