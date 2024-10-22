# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : raster.py
# Time       ：2024/9/10 11:22
# Author     ：author zhongq
# Description：raster toolbox
"""
import os
from osgeo import gdal
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
import rasterio
from rasterio import features
from shapely.geometry import shape
import geopandas as gpd
from app.utils import LOG, MetaInfoHandler, DetVisualizer
from app.core.settings import app_settings
SHARE_PATH = app_settings.share_path


class RSImage:
    def __init__(self, image, **kwargs):
        self.dataset = gdal.Open(image, gdal.GA_ReadOnly) if isinstance(
            image, str) else image
        assert isinstance(self.dataset, gdal.Dataset), \
            f'{image} is not a image'
        self._init_props(**kwargs)

    def _init_props(self, **kwargs):
        """Initialize properties."""
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.channel = self.dataset.RasterCount
        self.trans = self.dataset.GetGeoTransform()
        self.proj = self.dataset.GetProjection()
        self.band_list = [self.dataset.GetRasterBand(c + 1) for c in range(self.channel)]
        self.grids = []
        self.nodata = kwargs.get('nodata', [band.GetNoDataValue() for band in self.band_list])
        self.data_type = kwargs.get('data_type', self.band_list[0].DataType)
        self.meta = {
            'width': self.width,
            'height': self.height,
            'channel': self.channel,
            'trans': self.trans,
            'proj': self.proj,
            'data_type': self.data_type,
            'nodata': self.nodata

        }

    def read(self, grid: Optional[List] = None) -> np.ndarray:
        """Read image data. If grid is None, read the whole image.

        Args:
            grid (Optional[List], optional): Grid to read. Defaults to None.
        Returns:
            np.ndarray: Image data.
        """
        if grid is None:
            data = self.dataset.ReadAsArray()  # CHW
        else:
            assert len(
                grid) >= 4, 'grid must be a list containing at least 4 elements'
            data = self.dataset.ReadAsArray(grid[0], grid[1], grid[2] - grid[0], grid[3] - grid[1])
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        return data

    def write(self, data: Optional[np.ndarray], grid: Optional[List] = None):
        """Write image data.

        Args:
            grid (Optional[List], optional): Grid to write. Defaults to None. # xmin ymin xmax ymax.
            data (Optional[np.ndarray], optional): Data to write.
                Defaults to None.

        Raises:
            ValueError: Either grid or data must be provided.
        """

        assert data is not None
        if data.ndim != 3:
            raise ValueError("Image must have 3 dimensions (H, W, C) or (C, H, W)")

        if data.shape[2] == self.channel:
            # 如果图像格式为  (H, W, C) ，则转换为 (C, H, W)
            data = data.transpose(2, 0, 1)

        if data.shape[0] != self.channel:
            raise ValueError("Image must have 3 color channels")

        if grid is not None:
            for index, band in enumerate(self.band_list):
                if self.nodata[index] is not None:
                    band.SetNoDataValue(self.nodata[index])
                band.WriteArray(
                    data[index, :, :],
                    grid[0], grid[1])
        elif data is not None:
            for i in range(self.channel):
                if self.nodata[i] is not None:
                    self.band_list[i].SetNoDataValue(self.nodata[i])
                self.band_list[i].WriteArray(data[i])
        else:
            raise ValueError('Either grid or data must be provided.')

    def get_slice_bboxes(
            self,
            slice_width: Optional[int] = None,
            slice_height: Optional[int] = None,
            overlap_width_ratio: float = 0.2,
            overlap_height_ratio: float = 0.2,
    ) -> List[List[int]]:
        """Slices `image_pil` in crops.
        Corner values of each slice will be generated using the `slice_height`,
        `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

        Args:
            slice_height (int, optional): Height of each slice. Default None.
            slice_width (int, optional): Width of each slice. Default None.
            overlap_height_ratio(float): Fractional overlap in height of each
                slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
                overlap of 20 pixels). Default 0.2.
            overlap_width_ratio(float): Fractional overlap in width of each
                slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
                overlap of 20 pixels). Default 0.2.
            auto_slice_resolution (bool): if not set slice parameters such as slice_height and slice_width,
                it enables automatically calculate these params from image resolution and orientation.

        Returns:
            List[List[int]]: List of 4 corner coordinates for each N slices.
                [
                    [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                    ...
                    [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
                ]
        """
        slice_bboxes = []
        y_max = y_min = 0

        if slice_height and slice_width:
            y_overlap = int(overlap_height_ratio * slice_height)
            x_overlap = int(overlap_width_ratio * slice_width)
        # elif auto_slice_resolution:
        #     x_overlap, y_overlap, slice_width, slice_height = get_auto_slice_params(height=image_height, width=image_width)
        else:
            raise ValueError("Compute type is not auto and slice width and height are not provided.")

        while y_max < self.height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < self.width:
                x_max = x_min + slice_width
                if y_max > self.height or x_max > self.width:
                    xmax = min(self.width, x_max)
                    ymax = min(self.height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bboxes.append([xmin, ymin, xmax, ymax])
                else:
                    slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        return slice_bboxes

    def create_raster_file(self, raster_file: Path, gray: bool = False, grid: Optional[List] = None, **kwargs):

        width, height = (grid[2] - grid[0], grid[3] - grid[1]) if grid else (self.width, self.height)
        channel = 1 if gray else self.channel

        driver = gdal.GetDriverByName('GTiff')
        create_options = [
            'TILED=YES',
            'COMPRESS=DEFLATE',
            'PREDICTOR=2',
            'BLOCKXSIZE=256',
            'BLOCKYSIZE=256',
            'COPY_SRC_OVERVIEWS=YES'
        ]
        grid_image = driver.Create(raster_file.as_posix(), width, height, channel,
                                   self.data_type, options=create_options)
        grid_image.SetGeoTransform(self.trans)
        grid_image.SetProjection(self.proj)
        return RSImage(grid_image, **kwargs)


class RSGrids:
    def __init__(self, rs_image: RSImage, rs_file: Path, output_path: Optional[str] = None):
        self.rs_image = rs_image
        self.rs_file = rs_file
        self._prepare_path(output_path)
        self.meta = {
            "source_file": self.rs_file.relative_to(SHARE_PATH).as_posix(),
            "meta_path": self.meta_path.relative_to(SHARE_PATH).as_posix(),
            "grid_path": self.grids_path.relative_to(SHARE_PATH).as_posix()
        }

    def split_grids(self, window_size: Tuple[int, int], overlap_ratio: Tuple[float, float] = (0, 0)):
        self.grids = self.rs_image.get_slice_bboxes(*window_size, *overlap_ratio)
        kwargs = dict(nodata=self.rs_image.nodata, data_type=self.rs_image.data_type)
        for index, grid in enumerate(self.grids):
            data = self.rs_image.read(grid)  # c h  w
            dst_file = self.grids_path.joinpath('_'.join(map(str, grid)) + ".tif")
            grid_image = self.rs_image.create_raster_file(dst_file, gray=False, grid=grid, **kwargs)
            grid_image.write(data)
            grid_image.dataset.FlushCache()
            del grid_image

        self.meta["meta"] = self.rs_image.meta
        self.meta["grids"] = self.grids
        meta_file = MetaInfoHandler.write_metainfo(self.meta_path.joinpath("gridinfo.json"), self.meta)
        return {"meta_file": meta_file.relative_to(SHARE_PATH).as_posix()}

    @classmethod
    def merge_grids(cls, meta_file: str):
        meta_file = Path(SHARE_PATH).joinpath(meta_file)
        meta_info = MetaInfoHandler.parse_metainfo(meta_file)
        keys = ["source_file", "grids", "meta_path", "raster_path"]
        MetaInfoHandler.validate(meta_info, keys)

        source_file = Path(SHARE_PATH).joinpath(meta_info["source_file"])
        grids = meta_info["grids"]
        palette = meta_info.get("palette", None)
        meta_path = Path(SHARE_PATH).joinpath(meta_info["meta_path"])
        raster_path = Path(SHARE_PATH).joinpath(meta_info["raster_path"])
        source_image = RSImage(source_file.as_posix())

        dst_path = meta_path.joinpath("merge", source_file.name)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(nodata=source_image.nodata, data_type=source_image.data_type)
        merge_image = source_image.create_raster_file(dst_path, gray=True, grid=None, **kwargs)

        for grid in grids:
            infer_grid_path = raster_path.joinpath('_'.join(map(str, grid)) + ".tif")
            infer_grid_image = RSImage(infer_grid_path.as_posix())
            data = infer_grid_image.read()
            merge_image.write(data, grid)
        merge_image.dataset.FlushCache()
        del merge_image

        # color_table
        if palette is not None:
            color_map = {index: tuple(value) for index, value in enumerate(palette)}

            dst_color_path = meta_path.joinpath("color_merge", source_file.name)
            dst_color_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst_path.as_posix()) as src:
                shade = src.read(1)
                meta = src.meta
                with rasterio.open(dst_color_path.as_posix(), "w", **meta) as dst:
                    dst.write(shade, 1)
                    dst.write_colormap(1, color_map)
            meta_info.update(color_path=dst_color_path.relative_to(SHARE_PATH).as_posix())

        meta_info.update(merge_path=dst_path.relative_to(SHARE_PATH).as_posix())
        meta_file = MetaInfoHandler.write_metainfo(Path(meta_file), meta_info)
        return {"meta_file": meta_file.relative_to(SHARE_PATH).as_posix()}

    @classmethod
    def draw_det_instances(cls, meta_file: str, score_thr:float=0.3):
        meta_file = Path(SHARE_PATH).joinpath(meta_file)
        meta_info = MetaInfoHandler.parse_metainfo(meta_file)
        keys = ["source_file", "meta_path", "predinfo", "dataset_meta"]
        MetaInfoHandler.validate(meta_info, keys)
        source_file = Path(SHARE_PATH).joinpath(meta_info["source_file"])
        meta_path = Path(SHARE_PATH).joinpath(meta_info["meta_path"])
        pred_info = Path(SHARE_PATH).joinpath(meta_info["predinfo"])

        pred_instances = MetaInfoHandler.parse_metainfo(pred_info)
        pred_instances = pd.DataFrame(pred_instances)
        pred_instances = pred_instances[pred_instances["scores"] > score_thr]

        dataset_meta = meta_info["dataset_meta"]
        palette = [tuple(item) for item in dataset_meta["palette"]]
        classes = dataset_meta["classes"]


        rs_image = RSImage(source_file.as_posix())
        dst_path = meta_path.joinpath("visualizer", source_file.name)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(nodata=rs_image.nodata, data_type=rs_image.data_type)
        visual_image = rs_image.create_raster_file(dst_path, gray=False, **kwargs)

        rs_data = rs_image.read() # chw
        det_visualizer = DetVisualizer(rs_data, palette, classes)
        det_visualizer.visualize(pred_instances)
        visual_data = det_visualizer.get_image()  #hwc

        visual_image.write(visual_data)
        visual_image.dataset.FlushCache()
        del visual_image

        meta_info.update(visual_path=dst_path.relative_to(SHARE_PATH).as_posix())
        meta_file = MetaInfoHandler.write_metainfo(Path(meta_file), meta_info)
        return {"meta_file": meta_file.relative_to(SHARE_PATH).as_posix()}

    def _prepare_path(self, output_path):
        if output_path is None:
            output_path = self.rs_file.parent.joinpath(self.rs_file.stem, "grids")
        else:
            output_path = Path(output_path).joinpath(self.rs_file.stem, "grids")
        output_path.mkdir(parents=True, exist_ok=True)
        self.grids_path = output_path
        self.meta_path = output_path.parent

    @classmethod
    def from_image(cls, rs_file: str, output_path: Optional[str] = None):
        rs_file = Path(SHARE_PATH).joinpath(rs_file)
        if output_path:
            output_path = Path(SHARE_PATH).joinpath(output_path)
        if rs_file.is_file() and rs_file.suffix in ('.tif', '.tiff', '.TIFF', '.TIF'):
            return cls(
                RSImage(rs_file.as_posix()),
                rs_file,
                output_path)
        else:
            raise Exception('rs_image is not a valid image file')


async def raster_to_vector(meta_file: str, specific_label: Optional[str]=None):
    meta_file = Path(SHARE_PATH).joinpath(meta_file)
    meta_info = MetaInfoHandler.parse_metainfo(meta_file)
    keys = ["merge_path", "classes_map", "meta_path", "classes_map"]
    MetaInfoHandler.validate(meta_info, keys)
    merge_file = Path(SHARE_PATH).joinpath(meta_info["merge_path"])
    classes_map: dict = meta_info["classes_map"]
    meta_path = Path(SHARE_PATH).joinpath(meta_info["meta_path"])

    vector_path = meta_path.joinpath("vector", f"{merge_file.stem}.geojson")
    vector_path.parent.mkdir(parents=True, exist_ok=True)

    classes_map = {int(k): v for k, v in classes_map.items()}
    reverse_classes_map = {v: int(k) for k, v in classes_map.items()}
    specific_value = reverse_classes_map.get(specific_label, None) if reverse_classes_map else None
    max_value = len(classes_map)

    with rasterio.open(merge_file) as src:
        band = src.read(1)
        transform = src.transform
        meta = src.meta
        nodata = meta["nodata"]
        shapes = features.shapes(band, transform=transform)

        geoms, values = [], []
        for geom, value in shapes:
            if specific_value is not None and value != specific_value:
                continue
            # 如果没有指定 specific_label，检查是否要排除指定的值（例如背景）
            if specific_value is None:
                if (nodata is not None and
                        value == nodata):
                    continue
                if max_value and value >= max_value:
                    continue
            geoms.append(shape(geom))
            values.append(value)

        gdf = gpd.GeoDataFrame({'geometry': geoms, 'value': values})
        gdf.crs = meta['crs']
        gdf['label'] = gdf['value'].map(lambda v: classes_map.get(v, 'unknown'))
        gdf.to_file(vector_path.as_posix(), driver='GeoJSON')

    meta_info.update(vector_path=vector_path.relative_to(SHARE_PATH).as_posix())
    LOG.info(f"Raster has been polygonized and saved to {vector_path}")
    meta_file = MetaInfoHandler.write_metainfo(Path(meta_file), meta_info)
    return {"meta_file":meta_file.relative_to(SHARE_PATH).as_posix()}