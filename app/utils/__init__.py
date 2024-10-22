# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : __init__.py.py
# Time       ：2024/9/10 11:13
# Author     ：author zhongq
# Description：
"""
import json
from pathlib import Path
from typing import List
from app.utils.logging import Logger
from app.utils.Visualizer import DetVisualizer
LOG = Logger.logger()


class MetaInfoHandler:
    """A utility class for handling metadata information (JSON files)."""

    @staticmethod
    def write_metainfo(file_path: Path, data: dict) -> Path:
        """Write metadata information to a JSON file.

        Args:
            file_path (Path): The path where the JSON will be saved.
            data (dict): The metadata to be saved.

        Returns:
            Path: The path of the saved JSON file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        with file_path.open('w', newline='') as f:
            json.dump(data, f, indent=4)
        return file_path

    @staticmethod
    def parse_metainfo(file_path: Path) -> dict:
        """Parse metadata information from a JSON file.

        Args:
            file_path (str): The path of the JSON file to be read.

        Returns:
            dict: The parsed metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file path is invalid.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f'Meta file does not exist: {file_path}')
        with file_path.open() as f:
            return json.load(f)

    @staticmethod
    def validate(meta_info: dict, keys: List[str]):
        """
        Check if the metadata contains all the required keys.
        """
        for key in keys:
            value = meta_info.get(key)
            if value is None:
                raise ValueError(f'Meta info is missing key: {key}')