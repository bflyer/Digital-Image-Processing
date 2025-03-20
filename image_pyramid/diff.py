from PIL import Image
import numpy as np
import sys

def create_diff_image(img1_path, img2_path, output_path):
    """计算两张图片的像素级RGB差值并生成差异图"""
    # 读取图片并转换为RGB数组
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    
    arr1 = np.array(img1, dtype=np.int16)
    arr2 = np.array(img2, dtype=np.int16)

    # 验证尺寸一致性
    if arr1.shape != arr2.shape:
        raise ValueError("错误：图片尺寸不匹配")

    # 计算绝对差值并增强显示（可选）
    diff = np.abs(arr1 - arr2)

    import pdb; pdb.set_trace()
    # print(diff)
    
    # 增强对比度（差值放大2倍，可根据需要调整）
    enhanced_diff = np.clip(diff * 2, 0, 255).astype(np.uint8)
    
    # 生成并保存差异图
    Image.fromarray(enhanced_diff).save(output_path)
    print(f"✅ 差异图已保存至：{output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("使用方法：python diff.py <图片1> <图片2> <输出路径>")
        sys.exit(1)
    
    try:
        create_diff_image(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as e:
        print(f"❌ 错误：{str(e)}")

# python diff.py ./gaussian_pyramid/down_0.png ./reconstructed_gaussian_pyramid/recon_0.png ./diff_images/diff_0.png