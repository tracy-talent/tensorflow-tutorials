# -*- coding: utf-8 -*-
# @Time    : 19-1-24 下午7:32
# @Author  : liujian
# @File    : config.py
# @Software: PyCharm

import os

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

# 工程路径
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))

# 输入文件所在目录
INPUT_PATH = os.path.join(PROJECT_PATH, 'input')

# 输出文件所在目录
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'output')

# mnist数据集路径
MNIST_PATH = os.path.join(INPUT_PATH, 'MNIST_data')

# PTB数据集
PTB_PATH = os.path.join(INPUT_PATH, 'simple-examples/data')

