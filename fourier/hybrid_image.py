# import cv2
# import numpy as np

# def create_hybrid_image(image1_path, image2_path, output_path, 
#                         low_sigma=7, high_sigma=7, blend_ratio=0.5):
#     """
#     创建灰度混合图像
#     :param image1_path: 提供低频信息的图像路径
#     :param image2_path: 提供高频信息的图像路径
#     :param output_path: 输出图像保存路径
#     :param low_sigma: 低频高斯模糊强度（建议5-15）
#     :param high_sigma: 高频提取模糊强度（建议5-15）
#     :param blend_ratio: 高频混合比例（0.0-1.0）
#     """
#     # 读取图像并转为灰度
#     img_low = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
#     img_high = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
#     # 验证尺寸一致性
#     assert img_low.shape == img_high.shape, "图像尺寸必须相同"

#     # 低频处理：高斯模糊
#     low_freq = cv2.GaussianBlur(img_low, 
#                               (6*low_sigma+1, 6*low_sigma+1), 
#                               sigmaX=low_sigma)

#     # 高频处理：原始图像 - 模糊图像
#     blurred_high = cv2.GaussianBlur(img_high, 
#                                   (6*high_sigma+1, 6*high_sigma+1), 
#                                   sigmaX=high_sigma)
#     high_freq = img_high - blurred_high

#     # 混合图像（调整高频强度）
#     hybrid = low_freq + blend_ratio * high_freq

#     # 数值裁剪和类型转换
#     hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)

#     # 保存结果
#     cv2.imwrite(output_path, hybrid)
#     print(f"灰度混合图像已保存至 {output_path}")

#     # 显示效果预览（可选）
#     cv2.imshow('Hybrid (Close with ESC)', hybrid)
#     cv2.imshow('Low Frequency', low_freq.astype(np.uint8))
#     cv2.imshow('High Frequency', (high_freq + 128).astype(np.uint8))  # 可视化高频
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # 参数设置示例
#     create_hybrid_image(
#         image1_path="fourier\input\jiatong.jpg",   # 远距离可见的低频图像
#         image2_path="fourier\input\jiatong2.jpg",   # 近距离可见的高频图像
#         output_path="fourier\output\hybrid.jpg",
#         low_sigma=9,            # 增大该值会使低频更模糊
#         high_sigma=5,            # 增大该值会保留更多低频信息
#         blend_ratio=0.6          # 调整高频可见强度
#     )
import cv2
import numpy as np

def hybrid_image(low_img_path, high_img_path, sigma=5):
    """
    生成 hybrid image，要求两幅图像大小相同。
    
    参数:
        low_img_path: 用于提取低频信息的图像路径。
        high_img_path: 用于提取高频信息的图像路径。
        sigma: 高斯滤波器的标准差，值越大低通效果越明显。
        
    返回:
        hybrid: 混合图像（低频 + 高频）
    """

    # 读取两幅图像，并转换为灰度图
    low_img = cv2.imread(low_img_path, cv2.IMREAD_GRAYSCALE)
    high_img = cv2.imread(high_img_path, cv2.IMREAD_GRAYSCALE)

    if low_img is None or high_img is None:
        print("读取图像失败，请检查文件路径。")
        return None

    if low_img.shape != high_img.shape:
        print("两幅图像大小不一致，请确保尺寸相同。")
        return None

    # 1. 低频提取：
    # 对第一幅图像进行高斯低通滤波，保留低频信息
    low_frequencies = cv2.GaussianBlur(low_img, (0, 0), sigma)

    # 2. 高频提取：
    # 对第二幅图像进行高斯低通滤波，然后用原图减去低通结果得到高频分量
    high_lowpass = cv2.GaussianBlur(high_img, (0, 0), sigma)
    high_frequencies = cv2.subtract(high_img, high_lowpass)

    # 3. 合成 hybrid image：
    # 将低频图像与高频图像相加。注意，这里相加后的值可能需要裁剪到 0-255 的范围内。
    hybrid = cv2.add(low_frequencies, high_frequencies)

    return hybrid

def main():
    # 文件路径，根据需要进行修改
    low_img_path = "fourier\input\jiatong2.jpg"    # 用于低频提取的图像
    high_img_path = "fourier\input\jiatong.jpg"  # 用于高频提取的图像
    output_path = "fourier\output\hybrid_image2.jpg"
    
    # 调整 sigma 的值可改变低通/高通效果
    sigma = 5  
    
    hybrid = hybrid_image(low_img_path, high_img_path, sigma)
    if hybrid is not None:
        # 保存结果
        cv2.imwrite(output_path, hybrid)
        print("Hybrid image 已保存到:", output_path)
        
        # # 可选：显示生成的图像
        # cv2.imshow("Hybrid Image", hybrid)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("生成 hybrid image 失败。")

if __name__ == "__main__":
    main()
