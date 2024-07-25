#!/usr/bin/env python3

import os
import argparse
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def load_image_and_remove_transparency(image_path):
    image = Image.open(image_path)
    image = image.convert("RGBA")  # 确保图像是 RGBA 模式

    # 转换为 NumPy 数组
    data = np.array(image)
    # 只选择 Alpha 值大于 0 的像素
    non_transparent = data[:, :, 3] > 0
    data = data[non_transparent][:, :3]  # 剔除 Alpha 通道并保留非透明像素

    return data


def extract_colors(image_path, num_colors=10):
    print("Analyzing pic colors...")
    # 读取图像并转换为 RGB
    image = load_image_and_remove_transparency(image_path)
    data = np.array(image)

    # 重塑图像数组为（宽*高，3）
    reshaped_data = data.reshape(-1, 3)

    # 使用 KMeans 聚类算法
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(reshaped_data)

    # 获取聚类中心（主要颜色）
    colors = kmeans.cluster_centers_

    return colors, kmeans.labels_, reshaped_data


def plot_color_distribution(colors, labels, data, image_path, sample_rate=0.01):
    output_file = os.path.join(os.path.dirname(image_path), "color_distribution.png")

    print("Drawing distribution maps...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 生成采样索引
    indices = np.random.choice(
        data.shape[0], size=int(data.shape[0] * sample_rate), replace=False
    )

    # 对应聚类中心颜色的归一化
    cluster_colors = colors[labels] / 255.0

    # 绘制采样点，颜色使用对应的聚类中心颜色
    ax.scatter(
        data[indices, 0],
        data[indices, 1],
        data[indices, 2],
        c=cluster_colors[indices],
        marker="o",
        alpha=0.8,
        edgecolor=None,
    )

    ax.set_title("Color Clusters in RGB Space")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    plt.savefig(output_file)  # 保存图像到文件
    plt.close()  # 关闭图形，防止在内存中留下打开的窗口


def plot_color_proportions(colors, labels, image_path):
    output_file = os.path.join(os.path.dirname(image_path), "color_proportions.png")
    print("Drawing proportion maps...")
    # 计算每种颜色的占比
    unique, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()

    # 将 RGB 转换为 HSV
    hsv_colors = rgb_to_hsv(colors / 255.0)  # 先归一化RGB到[0, 1]范围

    # 根据色调（Hue）进行排序
    sorted_indices = np.argsort(hsv_colors[:, 0])  # HSV 中的第一个通道是色调
    sorted_colors = [colors[i] / 255 for i in sorted_indices]  # 归一化颜色
    sorted_proportions = proportions[sorted_indices]

    # 创建矩形图，颜色占比显示
    plt.figure(figsize=(10, 2), facecolor=None)
    current_position = 0
    for proportion, color in zip(sorted_proportions, sorted_colors):
        plt.fill_between(
            [current_position, current_position + proportion], 0, 1, color=color
        )
        current_position += proportion

    plt.xlim(0, 1)
    plt.axis("off")  # 关闭坐标轴
    plt.savefig(output_file, transparent=True)  # 保存到文件，背景透明
    plt.close()  # 关闭图形


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize image colors.")
    parser.add_argument("image_path", help="Path to the image file.")
    args = parser.parse_args()

    num_colors = 10  # Default number of colors
    colors, labels, reshaped_data = extract_colors(args.image_path, num_colors)

    plot_color_distribution(colors, labels, reshaped_data, args.image_path)
    plot_color_proportions(colors, labels, args.image_path)


if __name__ == "__main__":
    main()
