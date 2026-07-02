#!/usr/bin/env python3
"""
手写数字原图预处理 & 数据增强脚本
===========================================
功能:
1. 将 60 张 3456×3456 的 RGB 原图处理成 28×28 的 MNIST 格式灰度图
2. 对每张图做旋转、平移、缩放、加噪声等增强
3. 不删除原照片，结果保存到新文件夹
"""

import os
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import rotate, shift
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 崩溃
import matplotlib.pyplot as plt

# ======================== 路径配置 ========================
BASE_DIR = "/home/shorin/Projects/Py/大作业"
SRC_DIR = os.path.join(BASE_DIR, "手写数据_原图")
OUT_DIR = os.path.join(BASE_DIR, "手写数据_处理后")

# 输出子目录
OUT_ORIGINAL = os.path.join(OUT_DIR, "01_原始处理后")       # 28×28 原图（无增强）
OUT_ROTATED   = os.path.join(OUT_DIR, "02_旋转增强")
OUT_SHIFTED   = os.path.join(OUT_DIR, "03_平移增强")
OUT_SCALED    = os.path.join(OUT_DIR, "04_缩放增强")
OUT_NOISY     = os.path.join(OUT_DIR, "05_噪声增强")

# 确保输出子目录都存在
for d in [OUT_ORIGINAL, OUT_ROTATED, OUT_SHIFTED, OUT_SCALED, OUT_NOISY]:
    os.makedirs(d, exist_ok=True)

# ======================== 参数配置 ========================
TARGET_SIZE = 28                   # 输出图片尺寸
AUGMENT_TIMES = 5                  # 每种增强每张原图生成几张
ROTATION_RANGE = 15                # 旋转角度范围 ±15°
SHIFT_RANGE = 3                    # 平移像素范围 ±3
SCALE_RANGE = (0.9, 1.1)           # 缩放比例范围
NOISE_STRENGTH = 0.05              # 噪声强度


def load_and_preprocess(image_path):
    """
    加载单张图片 → 灰度化 → 缩放到 28×28 → 反色 → 归一化
    返回 numpy 数组 shape=(28,28)，像素值 0-255

    改进说明：
    - 原版用 img.point(lambda x: 0 if x < 80 else 255...) 三段式阈值，过于激进
    - 新版保留渐变信息：resize → 反色 → min-max 归一化 → 轻微 gamma 增强
    """
    img = Image.open(image_path).convert('L')          # 转灰度
    img = img.resize((TARGET_SIZE, TARGET_SIZE),
                     Image.LANCZOS)                     # 缩放到 28×28

    arr = np.array(img, dtype=np.float32)

    # 判断背景色——如果边缘大部分是浅色（白底），则反转
    edge_mean = (arr[0].mean() + arr[-1].mean() +
                 arr[:, 0].mean() + arr[:, -1].mean()) / 4
    if edge_mean > 128:
        arr = 255.0 - arr  # 白底黑字 → 黑底白字（MNIST 格式）

    # Min-max 归一化到 [0, 255]（保留渐变，不做硬阈值）
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0

    # 轻微 gamma 增强：压暗背景噪声，让笔画稍微更亮
    arr = 255.0 * (arr / 255.0) ** 0.9

    return np.clip(arr, 0, 255)


def apply_rotation(img_arr, angle_deg):
    """旋转增强"""
    rotated = rotate(img_arr, angle_deg, reshape=False, mode='nearest')
    return np.clip(rotated, 0, 255)


def apply_shift(img_arr, dx, dy):
    """平移增强"""
    shifted = shift(img_arr, shift=(dy, dx), mode='nearest')
    return np.clip(shifted, 0, 255)


def apply_scaling(img_arr, scale_factor):
    """缩放增强——缩放后重新填充/裁剪到 28×28"""
    from scipy.ndimage import zoom
    scaled = zoom(img_arr, scale_factor, order=1)
    h, w = scaled.shape
    out = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    if h > TARGET_SIZE:
        start = (h - TARGET_SIZE) // 2
        scaled = scaled[start:start+TARGET_SIZE, start:start+TARGET_SIZE]
    elif h < TARGET_SIZE:
        start = (TARGET_SIZE - h) // 2
        out[start:start+h, start:start+w] = scaled
        scaled = out
    return np.clip(scaled, 0, 255)


def apply_noise(img_arr, strength):
    """加椒盐噪声和高斯噪声混合"""
    arr = img_arr.copy()
    h, w = arr.shape

    # 高斯噪声
    gaussian = np.random.normal(0, strength * 255, (h, w))
    arr += gaussian

    # 椒盐噪声
    salt_pepper_prob = strength * 0.5
    rand = np.random.random((h, w))
    arr[rand < salt_pepper_prob / 2] = 0          # 椒
    arr[rand > 1 - salt_pepper_prob / 2] = 255    # 盐

    return np.clip(arr, 0, 255)


def save_image(arr, filepath):
    """保存 numpy 数组为 28×28 PNG 图片"""
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8, mode='L')
    img.save(filepath)


def main():
    # 读取所有图片文件
    image_files = sorted(
        f for f in os.listdir(SRC_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    print(f"找到 {len(image_files)} 张原图\n")

    total_augmented = 0

    for idx, fname in enumerate(image_files):
        src_path = os.path.join(SRC_DIR, fname)
        base_name = os.path.splitext(fname)[0]

        # ---------- Step 1: 预处理 ----------
        img_arr = load_and_preprocess(src_path)

        # 保存 28×28 原图
        save_image(img_arr, os.path.join(OUT_ORIGINAL, f"{base_name}.png"))

        # ---------- Step 2: 数据增强 ----------
        np.random.seed(hash(base_name) % (2**31))  # 每张图用固定随机种子保证可复现

        # 旋转增强
        for i in range(AUGMENT_TIMES):
            angle = np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
            aug = apply_rotation(img_arr, angle)
            save_image(aug, os.path.join(OUT_ROTATED, f"{base_name}_rot{i+1}.png"))
            total_augmented += 1

        # 平移增强
        for i in range(AUGMENT_TIMES):
            dx = np.random.randint(-SHIFT_RANGE, SHIFT_RANGE + 1)
            dy = np.random.randint(-SHIFT_RANGE, SHIFT_RANGE + 1)
            aug = apply_shift(img_arr, dx, dy)
            save_image(aug, os.path.join(OUT_SHIFTED, f"{base_name}_shift{i+1}.png"))
            total_augmented += 1

        # 缩放增强
        for i in range(AUGMENT_TIMES):
            sf = np.random.uniform(*SCALE_RANGE)
            aug = apply_scaling(img_arr, sf)
            save_image(aug, os.path.join(OUT_SCALED, f"{base_name}_scale{i+1}.png"))
            total_augmented += 1

        # 噪声增强
        for i in range(AUGMENT_TIMES):
            strength = np.random.uniform(0.02, NOISE_STRENGTH)
            aug = apply_noise(img_arr, strength)
            save_image(aug, os.path.join(OUT_NOISY, f"{base_name}_noisy{i+1}.png"))
            total_augmented += 1

        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx+1}/{len(image_files)} 张原图...")

    print(f"\n处理完成！")
    print(f"  原始处理后: {len(image_files)} 张 → {OUT_ORIGINAL}")
    print(f"  旋转增强:   {len(image_files) * AUGMENT_TIMES} 张 → {OUT_ROTATED}")
    print(f"  平移增强:   {len(image_files) * AUGMENT_TIMES} 张 → {OUT_SHIFTED}")
    print(f"  缩放增强:   {len(image_files) * AUGMENT_TIMES} 张 → {OUT_SCALED}")
    print(f"  噪声增强:   {len(image_files) * AUGMENT_TIMES} 张 → {OUT_NOISY}")
    print(f"  增强后总计: {len(image_files) + total_augmented} 张")

    # ---------- Step 3: 画对比预览图 ----------
    preview_files = sorted(os.listdir(OUT_ORIGINAL))[:5]
    fig, axes = plt.subplots(5, 6, figsize=(12, 10))
    for row, png in enumerate(preview_files):
        base = os.path.splitext(png)[0]
        # 原图
        axes[row, 0].imshow(
            np.array(Image.open(os.path.join(OUT_ORIGINAL, png))), cmap='gray')
        axes[row, 0].set_title("Original")
        axes[row, 0].axis('off')
        # 增强样本（找第一个对应增强）
        for col, (label, subdir) in enumerate([
            ("Rotated", OUT_ROTATED), ("Shifted", OUT_SHIFTED),
            ("Scaled", OUT_SCALED), ("Noisy", OUT_NOISY), ("---", OUT_NOISY)
        ], 1):
            if label == "---":
                aug_files = sorted([f for f in os.listdir(subdir) if base + "_" in f])
                if aug_files:
                    axes[row, col].imshow(
                        np.array(Image.open(os.path.join(subdir, aug_files[2]))),
                        cmap='gray')
                axes[row, col].set_title("Noisy(2)")
            else:
                aug_files = sorted([f for f in os.listdir(subdir) if base + "_" in f])
                if aug_files:
                    axes[row, col].imshow(
                        np.array(Image.open(os.path.join(subdir, aug_files[0]))),
                        cmap='gray')
                axes[row, col].set_title(label)
            axes[row, col].axis('off')

    plt.suptitle("Preprocessing + Augmentation Preview (First 5)", fontsize=14)
    plt.tight_layout()
    preview_path = os.path.join(OUT_DIR, "预览对比图.png")
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    print(f"\n预览图已保存: {preview_path}")
    plt.close()


if __name__ == "__main__":
    main()
