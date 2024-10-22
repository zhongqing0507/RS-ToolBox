# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Visualizer.py.py
# Time       ：2024/10/18 14:22
# Author     ：author zhongq
# Description：
"""

import numpy as np
from matplotlib.collections import PolyCollection
import warnings
from typing import Any, List, Optional, Tuple, Type, Union
from pandas import DataFrame


class DetVisualizer():

    def __init__(self,
                 image: np.ndarray,
                 palette: List[Tuple[int]],
                 classes: List[str],
                 line_width=2,
                 alpha=0.8,
                 text_color=(200, 200, 200)):
        self.palette = palette
        self.classes = classes
        self.line_width = line_width
        self.alpha = alpha
        self.text_color = text_color

        (self.fig_save_canvas, self.fig_save,
         self.ax_save) = self._initialize_fig()
        self.dpi = self.fig_save.get_dpi()
        if image is not None:
            self.set_image(image)

    def _initialize_fig(self) -> tuple:
        """Build figure according to fig_cfg.

        Args:
            fig_cfg (dict): The config to build figure.

        Returns:
             tuple: build canvas figure and axes.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(frameon=False)
        ax = fig.add_subplot()
        ax.axis(False)

        # remove white edges by set subplot margin
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        canvas = FigureCanvasAgg(fig)
        return canvas, fig, ax

    def set_image(self, image: np.ndarray) -> None:
        """Set the image to draw.

        Args:
            image (np.ndarray): The image to draw.
        """
        assert image is not None
        if image.ndim != 3:
            raise ValueError("Image must have 3 dimensions (H, W, C) or (C, H, W)")

        if image.shape[0] == 3:
            # 如果图像格式为 (3, H, W)，则转换为 (H, W, 3)
            image = image.transpose(1, 2, 0)

        if image.shape[2] != 3:
            raise ValueError("Image must have 3 color channels")

        image = image.astype('uint8')
        self._image = image
        self.width, self.height = image.shape[1], image.shape[0]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)

        # add a small 1e-2 to avoid precision lost due to matplotlib's
        # truncation (https://github.com/matplotlib/matplotlib/issues/15363)
        self.fig_save.set_size_inches(  # type: ignore
            (self.width + 1e-2) / self.dpi, (self.height + 1e-2) / self.dpi)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        self.ax_save.cla()
        self.ax_save.axis(False)
        self.ax_save.imshow(
            image,
            extent=(0, self.width, self.height, 0),
            interpolation='none')

    def visualize(self, pred_instance: DataFrame):
        # 计算图像宽高
        # image_height, image_width = image.shape[:2]

        # # 创建图像显示窗口
        # fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)  # 设置图像大小为原始尺寸
        # ax.imshow(image)
        if "bboxes" in pred_instance and "labels" in pred_instance:
            bboxes = np.array(pred_instance.bboxes.to_list())
            labels = np.array(pred_instance.labels.to_list())
        else:
            bboxes = []
            labels = []
            raise "No bboxes found in instance."

        # 处理bboxes和标签
        max_label = int(max(labels) if len(labels) > 0 else 0)

        # 获取颜色调色板
        bbox_palette = self.get_palette(self.palette, max_label + 1)
        colors = [bbox_palette[label] for label in labels]

        text_palette = self.get_palette(self.text_color, max_label + 1)
        text_colors = [text_palette[label] for label in labels]

        # 绘制bboxes
        self.draw_bboxes(
            bboxes,
            image_width=self.width,
            image_height=self.height,
            edge_colors=colors,
            alpha=self.alpha,
            line_widths=self.line_width
        )

        # 计算位置和面积
        positions = bboxes[:, :2] + self.line_width
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
        scales = self._get_adaptive_scales(areas)

        for i, (pos, label) in enumerate(zip(positions, labels)):
            # 获取标签文本
            label_text = self.classes[label + 1] if self.classes is not None else f'class {label}'

            # 如果有分数，添加到标签文本
            if 'scores' in pred_instance:
                score = round(float(pred_instance.scores[i]) * 100, 1)
                label_text += f': {score}'
            # 绘制文本
            self.draw_texts(
                label_text,
                pos,
                font_sizes=[int(13 * scales[i])],  # 自适应字体大小
                colors=[text_colors[i]],
                bboxes=[{
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                }]
            )

        # # 保存图像到指定路径
        # if save_path is not None:
        #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        #     plt.close(fig)  # 关闭图形以释放内存
        # else:
        #     plt.show()  # 显示图形（如果未提供保存路径）

    def draw_bboxes(self,
                    bboxes: np.ndarray,
                    image_width: int,
                    image_height: int,
                    edge_colors: Union[str, tuple, list] = 'g',
                    line_styles: Union[str, list] = '-',
                    line_widths: Union[float, list] = 2,
                    face_colors: Union[str, tuple, list] = 'none',
                    alpha: float = 0.8,
                    ax=None  # matplotlib的axes
                    ):
        """Draw single or multiple bboxes."""

        if len(bboxes.shape) == 1:
            bboxes = bboxes[None]
        assert bboxes.shape[-1] == 4, f'The shape of `bboxes` should be (N, 4), but got {bboxes.shape}'

        assert (bboxes[:, 0] <= bboxes[:, 2]).all() and (bboxes[:, 1] <= bboxes[:, 3]).all()

        if not self._is_posion_valid(bboxes.reshape((-1, 2, 2)), image_width, image_height):
            warnings.warn('Warning: The bbox is out of bounds, the drawn bbox may not be in the image', UserWarning)

        # 转换bboxes为多边形
        poly = np.stack(
            (bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 1],
             bboxes[:, 2], bboxes[:, 3], bboxes[:, 0], bboxes[:, 3]),
            axis=-1).reshape(-1, 4, 2)
        poly = [p for p in poly]

        return self.draw_polygons(
            poly,
            image_width=image_width,
            image_height=image_height,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=face_colors
        )

    def draw_polygons(self,
                      polygons: Union[np.ndarray, list],
                      image_width: int,
                      image_height: int,
                      edge_colors: Union[str, tuple, list] = 'g',
                      line_styles: Union[str, list] = '-',
                      line_widths: Union[float, list] = 2,
                      face_colors: Union[str, tuple, list] = 'none',
                      alpha: float = 0.8
                      ):
        """Draw single or multiple polygons."""
        self.check_type('polygons', polygons, (list, np.ndarray))
        edge_colors = self.color_val_matplotlib(edge_colors)
        face_colors = self.color_val_matplotlib(face_colors)

        # polygons = [tensor2ndarray(polygon) for polygon in polygons]
        for polygon in polygons:
            if not self._is_posion_valid(polygon, image_width, image_height):
                warnings.warn('Warning: The polygon is out of bounds, the drawn polygon may not be in the image',
                              UserWarning)

        if isinstance(line_widths, (int, float)):
            line_widths = [line_widths] * len(polygons)
        line_widths = [min(max(linewidth, 1), 10) for linewidth in line_widths]

        polygon_collection = PolyCollection(
            polygons,
            alpha=alpha,
            facecolor=face_colors,
            linestyles=line_styles,
            edgecolors=edge_colors,
            linewidths=line_widths
        )

        self.ax_save.add_collection(polygon_collection)

    def draw_texts(self,
                   texts: Union[str, List[str]],
                   positions: np.ndarray,
                   font_sizes: Optional[Union[int, List[int]]] = None,
                   colors: Union[str, tuple, List[str], List[tuple]] = 'g',
                   bboxes: Optional[Union[dict, List[dict]]] = None
                   ):
        """Draw single or multiple text boxes."""
        # from matplotlib.font_manager import FontProperties

        self.check_type('texts', texts, (str, list))
        if isinstance(texts, str):
            texts = [texts]

        num_text = len(texts)

        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape == (num_text, 2), (
            '`positions` should have the shape of '
            f'({num_text}, 2), but got {positions.shape}')

        colors = self.color_val_matplotlib(colors)

        if bboxes is None:
            bboxes = [None for _ in range(num_text)]

        for i in range(num_text):
            self.ax_save.text(
                positions[i][0],
                positions[i][1],
                texts[i],
                size=font_sizes[i] if font_sizes else 10,
                bbox=bboxes[i],
                color=colors[i] if isinstance(colors, list) else colors,
                verticalalignment='top',
                horizontalalignment='left'
            )

    def get_palette(self, palette: Union[List[tuple], str, tuple],
                    num_classes: int) -> List[Tuple[int]]:
        assert isinstance(num_classes, int)
        if isinstance(palette, list):
            dataset_palette = palette
        elif isinstance(palette, tuple):
            dataset_palette = [palette] * num_classes
        return dataset_palette

    def _is_posion_valid(self, position: np.ndarray, width, height) -> bool:
        """Judge whether the position is in image.

        Args:
            position (np.ndarray): The position to judge which last dim must
                be two and the format is [x, y].

        Returns:
            bool: Whether the position is in image.
        """
        flag = (position[..., 0] < width).all() and \
               (position[..., 0] >= 0).all() and \
               (position[..., 1] < height).all() and \
               (position[..., 1] >= 0).all()
        return flag

    def check_type(self, name: str, value: Any,
                   valid_type: Union[Type, Tuple[Type, ...]]) -> None:
        """Check whether the type of value is in ``valid_type``.

        Args:
            name (str): value name.
            value (Any): value.
            valid_type (Type, Tuple[Type, ...]): expected type.
        """
        if not isinstance(value, valid_type):
            raise TypeError(f'`{name}` should be {valid_type} '
                            f' but got {type(value)}')

    def _get_adaptive_scales(self, areas: np.ndarray,
                             min_area: int = 800,
                             max_area: int = 30000) -> np.ndarray:
        """Get adaptive scales according to areas.

        The scale range is [0.5, 1.0]. When the area is less than
        ``min_area``, the scale is 0.5 while the area is larger than
        ``max_area``, the scale is 1.0.

        Args:
            areas (ndarray): The areas of bboxes or masks with the
                shape of (n, ).
            min_area (int): Lower bound areas for adaptive scales.
                Defaults to 800.
            max_area (int): Upper bound areas for adaptive scales.
                Defaults to 30000.

        Returns:
            ndarray: The adaotive scales with the shape of (n, ).
        """
        scales = 0.5 + (areas - min_area) // (max_area - min_area)
        scales = np.clip(scales, 0.5, 1.0)
        return scales

    def color_val_matplotlib(self,
                             colors: Union[str, tuple, List[Union[str, tuple]]]
                             ) -> Union[str, tuple, List[Union[str, tuple]]]:
        """Convert various input in RGB order to normalized RGB matplotlib color
        tuples,
        Args:
            colors (Union[str, tuple, List[Union[str, tuple]]]): Color inputs
        Returns:
            Union[str, tuple, List[Union[str, tuple]]]: A tuple of 3 normalized
            floats indicating RGB channels.
        """
        if isinstance(colors, str):
            return colors
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            for channel in colors:
                assert 0 <= channel <= 255
            colors = [channel / 255 for channel in colors]
            return tuple(colors)
        elif isinstance(colors, list):
            colors = [
                self.color_val_matplotlib(color)  # type:ignore
                for color in colors
            ]
            return colors
        else:
            raise TypeError(f'Invalid type for color: {type(colors)}')

    def get_image(self) -> np.ndarray:
        """Get RGB image from ``FigureCanvasAgg``.

        Args:
            canvas (FigureCanvasAgg): The canvas to get image.

        Returns:
            np.ndarray: the output of image in RGB.
        """
        assert self._image is not None, 'Please set image using `set_image`'
        s, (width, height) = self.fig_save_canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype('uint8')


# # # 使用示例
# import os.path as osp
# import matplotlib.image as mpimg
# import json
# import pandas as pd
#
# score_thr = 0.3
# root_path = "/mnt/tempdata/user_temp_data/a09179de-3c62-4ba6-8ed6-75b59e42280a"
# out_dir = "/mnt/tempdata/user_temp_data/a09179de-3c62-4ba6-8ed6-75b59e42280a/plane_08"
# predinfo = osp.join(out_dir, "predinfo.json")
# with open(predinfo, "r") as f:
#     predinfo = json.load(f)
# pred_df = pd.DataFrame(predinfo)
# pred_df = pred_df[pred_df["scores"] > score_thr]
#
# gridinfo = osp.join(out_dir, "gridinfo.json")
# with open(gridinfo, "r") as f:
#     gridinfo = json.load(f)
#
# dataset_meta = gridinfo["dataset_meta"]
# palette = [tuple(item) for item in dataset_meta["palette"]]
# classes = dataset_meta["classes"]
#
# image_path = osp.join(root_path, "plane_08.tif")
# image = mpimg.imread(image_path)  # 生成示例图片 # HWC
#
# det_visualizer = DetVisualizer(image, palette, classes)
# det_visualizer.visualize(pred_df)
# result_image = det_visualizer.get_image()
#
# print(result_image.shape)
# from PIL import Image
# image_pil = Image.fromarray(result_image)  # HWC
#
# # Save the image as PNG
# image_pil.save("./test_2.png", format='PNG')
# print("1111")
