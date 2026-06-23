#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# 原始数据

def soft_label():
    # read TXT FILE from /home/student/Documents/data/CholecT45/dict/maps.txt
    with open('/mnt/ssd/data/CholecT45/dict/maps.txt', 'r') as f:
        data = f.read()
    # sort data according to the first column, except the first row
    header, *rows = data.splitlines()
    rows = sorted(rows, key=lambda x: int(x.split(",")[0]))
    data = "\n".join([header] + rows)

    # 读取数据
    df = pd.read_csv(StringIO(data))

    # 提取第二列（I列）
    I_values = df[' I'].values

    # 生成混淆矩阵
    confusion_matrix_I = (I_values[:, None] == I_values[None, :]).astype(int)

    # 提取第二列（I列）
    V_values = df[' V'].values

    # 生成混淆矩阵
    confusion_matrix_V = (V_values[:, None] == V_values[None, :]).astype(int)

    T_values = df[' T'].values
    confusion_matrix_T = (T_values[:, None] == T_values[None, :]).astype(int)

    #confusion_matrix = (confusion_matrix_I + confusion_matrix_V + confusion_matrix_T) / 3.0
    confusion_matrix = (confusion_matrix_I + confusion_matrix_V + confusion_matrix_T) / 3.0
    return confusion_matrix

    #绘制热力图
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(confusion_matrix,
    #             cmap='Reds',  # 使用蓝色渐变，数值越大颜色越深
    #             annot=False,   # 不显示数值
    #             square=True,   # 保持正方形
    #             cbar=True)    # 不显示颜色条

    # plt.title('Confusion Matrix Based on IVT Values')
    # plt.xlabel('Class Sorted')
    # plt.ylabel('Class Sorted')
    # plt.savefig('ivt.png', dpi=300, bbox_inches='tight')
    # plt.show()
soft_label()