import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 1. 设定通用梯度配色 (根据您提供的配色方案)
# #FFCF36 (金黄), #9BC34A (嫩绿), #40AB6C (绿), #008C7F (青), #006A78 (深青), #2F4858 (深灰)
colors = ["#FFCF36", "#9BC34A", "#40AB6C", "#008C7F", "#006A78", "#2F4858"]
# 逆序排列：让高 K 值区域显示为黄色，背景低值区为深灰色
cmap_custom = LinearSegmentedColormap.from_list("custom_gradient", colors[::-1])

# 2. 建立空间网格 (单位: 米)
# 以广州塔为原点 (0,0)，设定计算域范围
x_lim, y_lim = 1000, 1000
res = 500  # 网格分辨率
x = np.linspace(-x_lim, x_lim, res)
y = np.linspace(-y_lim, y_lim, res)
X, Y = np.meshgrid(x, y)

# 3. 抽象周围建筑位置关系 (xi, yi, Ai, Ri)
# Ai: 增强强度, Ri: 影响半径
buildings = [
    {"name": "Media Port", "x": -420, "y": 80, "A": 2.5, "R": 150},
    {"name": "PR North", "x": 480, "y": 220, "A": 2.0, "R": 120},
    {"name": "PR East", "x": 550, "y": -120, "A": 2.2, "R": 180},
    {"name": "Boyawan", "x": -280, "y": -380, "A": 1.8, "R": 130},
    {"name": "Dijing Huayuan", "x": -550, "y": -250, "A": 2.1, "R": 140}
]

# 4. 计算建筑增强函数 Fbld(x, y) [引用自模型 2.2]
Fbld = np.ones_like(X)
for b in buildings:
    # 计算到各个建筑中心的欧几里得距离的平方
    dist_sq = (X - b['x'])**2 + (Y - b['y'])**2
    # 高斯凸起累加
    Fbld += b['A'] * np.exp(-dist_sq / b['R']**2)

# 5. 计算风向加权后的扩散系数 [引用自模型 2.3]
K0 = 1.0                # 背景水平扩散强度
theta = np.deg2rad(45)  # 设定主导风向角 (例如 45度)
alpha = 0.6             # 侧风扩散削弱系数 (alpha < 1 表示顺风方向扩散更强)

Kx = K0 * (np.cos(theta)**2 + alpha * np.sin(theta)**2) * Fbld
Ky = K0 * (np.sin(theta)**2 + alpha * np.cos(theta)**2) * Fbld

# 计算总水平扩散能力 (模长)
K_total = np.sqrt(Kx**2 + Ky**2)

# 6. 可视化 K 的空间分布
plt.figure(figsize=(12, 10), dpi=100)
plt.style.use('dark_background') # 使用深色背景提升对比度

# 绘制热力图
mesh = plt.pcolormesh(X, Y, K_total, shading='auto', cmap=cmap_custom)
cbar = plt.colorbar(mesh)
cbar.set_label('Turbulent Diffusion Coefficient $K_{total}$', fontsize=12)

# 标注广州塔位置
plt.scatter(0, 0, marker='*', color='#FFCF36', s=300, edgecolors='white', label='Canton Tower')

# 标注建筑群中心
for b in buildings:
    plt.text(b['x'], b['y']+40, b['name'], color='white', ha='center', fontsize=9, fontweight='bold')

plt.title('Spatial Distribution of $K$ (Urban Diffusion Map)', fontsize=15, pad=20)
plt.xlabel('Distance West - East (m)')
plt.ylabel('Distance South - North (m)')
plt.grid(alpha=0.1)
plt.legend()

plt.show()