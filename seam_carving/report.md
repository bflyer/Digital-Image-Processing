## Seam Carving
### Image Shrinking
#### 核心思路
1. 我们认为能量低的区域对于视觉上的影响不大，因此我们希望优先删除能量低的区域。能量有许多定义方式，在我的代码中使用 x 和 y 的梯度绝对值之和作为能量
2. 为了尽可能地减小能量并保持图像连续（不是直接选择竖线或横线），但又保持语义（不是直接选择能量最低的像素），因此我们选择一次删除一条连续的 seam。
3. 搜索 seam 时我们使用动态规划，先自顶向下更新 cost map 和 path map，然后沿着 path map 回溯获得 seam。

#### 踩过的坑
1. 回溯获得 seam 得到的是从末端到顶端的列表，返回时需要 reverse 一下。

### Image Enlarging
#### 核心思路
直观上理解，能量低的 seam 多了还是少了视觉上影响不大，Image Shrinking 的时候我们是把无关痛痒的删掉，而 Image Enlarging 的时候我们是填充一些无关痛痒的 seam。
1. 一次性找到 k 条能量最低的 seam
2. 用原图中这些 seam 左右的像素对其进行填充

#### 踩过的坑
放大比起缩小的坑简直是多得多得多……
坑 1. 我们需要一次性获得原图中能量最低的 k 条数据，而不是逐条获取，否则就会导致重复选取能量最低的那条，导致视觉重复。

坑 2. 我最直觉的做法是，既然我们不能重复选取，那选一条就删一条呗，但尝试之后发现这并不可行。一方面，删除导致排序混乱，需要大量的修补措施；另一方面，删除导致一条原图中不存在的 seam 可能因为删除而在新图中出现，而我希望这些 seam 在原图中就不重叠，和缩小是不同的。

坑 3. 之后我改为将已经选中的 seam 的 energy 赋为 inf，并且在回溯 seam 的时候检测，如果这条 seam 上存在 inf，那么它就和之前的重叠了，我会把这条路径上的 energy 全部赋为 inf 并且丢弃这条 seam 重新来过；此外，如果无法选出新的 seam 了，那么我就循环使用之前找出来的所有 seam。之所以复用低能量区域是为了尽量避免放大原图中高能量的主体，从而保持主体的一致性。

坑 4. 另一个大坑是处理编号问题，因为在我们插入的过程中，新图会不断扩大，因此原来的编号可能需要更新，但问题在于，这些 seam 的顺序是不确定的。以横向扩展为例，一个中间的 seam 可能先插入，这对于它左侧的 seam 没有影响，但会导致它右侧的 seam 需要编号加 1，不能一概而论。于是我就想到反正这些 seam 主要依赖于原图的像素，插入顺序影响不大，因此我就按从小到大的顺序重新排了一边，保证先插入的 seam 在最左侧，之后每一条 seam 的 index 要加上“它的轮数-1”，这样就解决了。

坑 5. 在已经获得 top k 条 seams、添加这些 seams 时，我发现最后的图像总会出现一些裂缝。比较幸运的是第一条 seam 就是裂缝，于是我将其值以及组成它的相邻像素打印出来，发现因为 image 是 uint8 类型，而我在加和之后才取平均，导致了溢出。因此修改就是分别取平均再相加。修改前后效果如下图：
<div style="display: flex; justify-content: space-between;">
  <img src="output/rider_overflow.png" alt="图片1描述" style="width: 48%;" />
  <img src="output/rider-+.png" alt="图片2描述" style="width: 48%;" />
</div>

## Object Removal
#### 核心思路
1. 将要去除的 mask 区域能量值设低，要保护的 mask 区域能量值设高，先进行 Image Shrinking 删去包含 mask 的 seams
2. 然后进行 Image Enlarging 恢复图像大小。

### Mask 生成
我编写了交互式的 mask 生成脚本，使用说明如下：
1. **窗口说明**  
   - 窗口标题：`Object Removal - Draw Mask`（物体移除 - 绘制掩膜）
   - 实时显示原始图像与掩膜叠加效果（掩膜以蓝色半透明显示）。

2. **鼠标操作**  
   - **左键拖动**：绘制掩膜（标记待删除物体）。  
   - **右键拖动**：擦除已绘制的掩膜。  

3. **键盘快捷键**  
   | 按键 | 功能 |  
   |------|------|  
   | `+`  | 增大画笔尺寸 |（注意 '+' 需要按住 Ctrl 键）  
   | `-`  | 减小画笔尺寸 |  
   | `m`  | 切换绘制/擦除模式 |  
   | `s`  | 保存掩膜并退出 |  
   | `q`  | 放弃保存直接退出 |  

4. **界面提示**  
   - 左上角显示当前画笔大小（`Brush: XX`）。  
   - 左上角显示当前模式（`Mode: Draw` 或 `Mode: Erase`）。  

#### 踩过的坑
1. 当同时有保护和去除的 mask 时，若两个 mask 重叠，那么可能导致要去除的 mask 的部分区域始终处于一个比较高的值，从而导致难以去除，会伤及大量其他能量低的区域。我一开始是：  
`modified_energy = energy - (mask_remove * 1e6) + (mask_preserve * 1e6)`，  
就会出现上述情况。
![alt text](output/intersection.png)![alt text](output/couple_middle_failure.png) ![alt text](output/couple_failure.png) 
改进的方法是：优先要去除的 mask，所以降低 `mask_preserve` 的权重，改为：  
`modified_energy = energy - (mask_remove * 1e6) + (mask_preserve * 1e4)`，  
这样就可以优先去除啦。效果如图：
![alt text](output/couple_middle.png) ![alt text](output/couple.png)