import numpy as np
import cv2

# 读取两张二值图像（确保尺寸相同）
image1 = cv2.imread('seam_carving\mask\mask_preserve.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('seam_carving\mask\mask_remove.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否加载成功
assert image1 is not None and image2 is not None, "图像加载失败，请检查路径"
assert image1.shape == image2.shape, "图像尺寸必须相同"

# 转换为布尔型（255 -> True, 0 -> False）并计算交集
intersection = (image1 == 255) & (image2 == 255)

# 将结果转回0/255格式（True->255, False->0）
result = np.where(intersection, 255, 0).astype(np.uint8)

# 保存结果
cv2.imwrite('intersection.png', result)

# 可视化（可选）
cv2.imshow('Intersection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()