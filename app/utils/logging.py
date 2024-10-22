# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : logging.py
# Time       ：2024/9/10 11:13
# Author     ：author zhongq
# Description：
"""
import logging
from pathlib import Path


def prepare_folder(root_folder: Path):
    if isinstance(root_folder, str):
        root_folder = Path(root_folder)
    root_folder.mkdir(parents=True, exist_ok=True)
    return root_folder


class Logger:
    def __init__(self, log_file_path='/home/logs/app.log'):
        self.log_file_path = log_file_path
        self.logger = self._create_logger()

    def _create_logger(self):
        # 准备目录
        prepare_folder(Path(self.log_file_path).parent)
        # 创建日志记录器
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # 创建文件处理程序
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # 创建控制台处理程序
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将处理程序添加到记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    @staticmethod
    def logger():
        return Logger()


# 示例使用
if __name__ == "__main__":
    # 创建日志记录组件
    log = Logger.logger()

    # 记录不同级别的日志
    log.info("This is an info message")
    log.debug("This is a debug message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.critical("This is a critical message")
