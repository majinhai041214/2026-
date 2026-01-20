import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 1. 采用你提供的通用梯度配色
color_hex = ['#FFCF36', '#9BC34A', '#40AB6C', '#008C7F', '#006A78', '#2F4858']
cmap_custom = LinearSegmentedColormap.from_list('custom', color_hex[::-1])

# 2. 建筑抽象参数 (xi, yi, Ai, Ri)
buildings = [
    {'name': 'Media Port', 'x': -420, 'y': 80, 'A': 2.5, 'R': 150},      # 媒体港
    {'name': 'PR North', 'x': 480, 'y': 220, 'A': 2.0, 'R': 120},       # 珠江帝景北
    {'name': 'PR East', 'x': 550, 'y': -120, 'A': 2.2, 'R': 180},      # 珠江帝景东
    {'name': 'Boyawan', 'x': -280, 'y': -380, 'A': 1.8, 'R': 130},     # 泊雅湾/利安花园
    {'name': 'Dijing Huayuan', 'x': -550, 'y': -250, 'A': 2.1, 'R': 140} # 帝景华苑
]

# 3. 生成 F_bld 场
x = np.linspace(-1000, 1000, 400)
y = np.linspace(-1000, 1000, 400)
X, Y = np.meshgrid(x, y)
Fbld = np.ones_like(X)

for b in buildings:
    Fbld += b['A'] * np.exp(-((X - b['x'])**2 + (Y - b['y'])**2) / b['R']**2)

# 4. 绘图
plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Fbld, levels=50, cmap=cmap_custom)
plt.colorbar(label='Building Enhancement Factor $F_{bld}$')

# 标注广州塔和建筑中心
plt.plot(0, 0, 'r*', markersize=15, label='Canton Tower (Source)')
for b in buildings:
    plt.plot(b['x'], b['y'], 'wo', markersize=4)
    plt.text(b['x']+20, b['y']+20, b['name'], color='white', fontsize=9, fontweight='bold')

plt.title('Abstract Urban Canopy Model: $F_{bld}(x, y)$ Field', fontsize=14)
plt.xlabel('West - East (m)')
plt.ylabel('South - North (m)')
plt.grid(alpha=0.2)
plt.legend()
plt.show()