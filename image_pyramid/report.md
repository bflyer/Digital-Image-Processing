## image_reconstruction

## 踩过的坑
1. **重建图片亮度过高**
**现象描述**：初版执行任务一“image reconstruction”时，得到的图片亮度过高。<br>
**问题分析**：经过检查发现是因为我们在进行图像处理的时候，为了避免卷积等操作的溢出，选择先将图像归一化至 `[0, 1]`，而在计算 laplacian pyramid 时，可能得到负值，而这个负值在还原回 `[0, 255]` 时被迫变成了 0，因此导致最后亮度偏高。<br>  
**解决办法**：由于下采样再上采样之后的像素值并不会差太多，所以我们可以认为 laplacian pyramid 的像素值会落在 `[-0.5, 0.5]` 这个区间。
因此我们在保存的时候就可以先将其加上一个 offset 0.5，然后再保存；要用的时候则是加载完减去 offset，这样就成功保留了负值。
<br>
2. **重建图片局部偏暗**
**现象描述**：修改之后如果取 gaussian pyramid top 作为重建的 top 则能成功重建，但如果选择 top 为 laplacian pyramid 的 top，则重建的图片有一部分亮度过低，可但理论上二者是同一个东西，不应该有所差异。<br>
**问题分析**：经过检查发现是因为我们在改进 laplacian 保存方法时，我们是假设 laplacian pyramid 的像素值会落在 `[-0.5, 0.5]` 这个区间，
而这个假设的前提是 laplacian pyramid 是由相邻 gaussian pyramid level 作差得到的，可是 laplacian pyramid top 是直接复制 gaussian pyramid top，
因此完全可能超出我们假设的区间。<br>
**解决办法**：我们只需要分类讨论，在保存 laplacian pyramid top 时正常报错，加载时正常加载就行。


