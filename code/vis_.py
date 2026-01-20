import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ================= 1. 环境与参数初始化 =================
# 配色方案 (基于)
color_list = ["#FFCF36", "#9BC34A", "#40AB6C", "#008C7F", "#006A78", "#2F4858"]
cmap_custom = LinearSegmentedColormap.from_list("custom", color_list[::-1])

dx, dy, dz = 20, 20, 10
nx, ny, nz = 101, 101, 71
dt = 3
total_time = 2400
steps = int(total_time / dt)

x_vec = np.linspace(-1000, 1000, nx)
y_vec = np.linspace(-1000, 1000, ny)
z_vec = np.linspace(0, 700, nz)
# 创建 3D 网格坐标
X_3d, Y_3d, Z_3d = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')

# ================= 2. 物理场构建 (基于) =================
# 陆家嘴建筑群数据
buildings_data = [
    {"name": "Aquarium", "x": 210, "y": 50, "z": 30, "w": 80},
    {"name": "Convention", "x": -100, "y": 85, "z": 55, "w": 100},
    {"name": "Mall", "x": -95, "y": -385, "z": 50, "w": 120},
    {"name": "IFC", "x": 195, "y": -410, "z": 255, "w": 90},
    {"name": "BankOfChina", "x": 335, "y": -160, "z": 226, "w": 80},
    {"name": "JinMao", "x": 610, "y": -530, "z": 420, "w": 80},
    {"name": "SWFC", "x": 780, "y": -540, "z": 492, "w": 80},
    {"name": "ShanghaiTower", "x": 650, "y": -720, "z": 632, "w": 100}
]

# 模拟之前的扩散系数逻辑
X_2d, Y_2d = np.meshgrid(x_vec, y_vec, indexing='ij')
Fbld = np.ones((nx, ny))
for b in buildings_data:
    Fbld += 2.0 * np.exp(-((X_2d - b['x'])**2 + (Y_2d - b['y'])**2) / 150**2)

K0, theta, alpha = 15.0, np.deg2rad(80), 0.4 
Kx_2d = K0 * (np.cos(theta)**2 + alpha * np.sin(theta)**2) * Fbld
Ky_2d = K0 * (np.sin(theta)**2 + alpha * np.cos(theta)**2) * Fbld
Kz_vec = 2.0 * (1 - np.exp(-z_vec / 20.0))

Kx = np.repeat(Kx_2d[:, :, np.newaxis], nz, axis=2)
Ky = np.repeat(Ky_2d[:, :, np.newaxis], nz, axis=2)
Kz = np.tile(Kz_vec, (nx, ny, 1))

# ================= 3. 模拟计算 =================
C = np.zeros((nx, ny, nz))
M = 1e20 
C[nx//2, ny//2, int(468/dz)] = M / (dx*dy*dz) # 东方明珠高度 468m

for s in range(steps):
    C_old = C.copy()
    C[1:-1, 1:-1, 1:-1] += dt * (
        (Kx[1:-1, 1:-1, 1:-1] * (C_old[2:, 1:-1, 1:-1] - 2*C_old[1:-1, 1:-1, 1:-1] + C_old[:-2, 1:-1, 1:-1])) / dx**2 +
        (Ky[1:-1, 1:-1, 1:-1] * (C_old[1:-1, 2:, 1:-1] - 2*C_old[1:-1, 1:-1, 1:-1] + C_old[1:-1, :-2, 1:-1])) / dy**2 +
        (Kz[1:-1, 1:-1, 1:-1] * (C_old[1:-1, 1:-1, 2:] - 2*C_old[1:-1, 1:-1, 1:-1] + C_old[1:-1, 1:-1, :-2])) / dz**2
    )
    C[:, :, 0] = C[:, :, 1]
    C[:, :, -1] = C[:, :, -2]

# ================= 4. 3D 点云可视化模块 =================
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#f0f0f0') # 浅灰色背景

# --- A. 过滤浓度点云 ---
# 只显示浓度大于最高值 10^-2 倍的点，避免杂讯
threshold = C.max() * 0.1
mask = C > threshold

# 提取绘图数据
x_p, y_p, z_p = X_3d[mask], Y_3d[mask], Z_3d[mask]
c_p = C[mask]

# 绘制点云：深色代表高浓度，点的大小随浓度变化
# 使用对数尺度映射颜色
scatter = ax.scatter(x_p, y_p, z_p, c=c_p, 
                    norm=LogNorm(vmin=threshold, vmax=C.max()),
                    cmap=cmap_custom,
                    s=np.log10(c_p/threshold)*5 + 1, # 浓度越高点越大
                    alpha=0.15, # 极低透明度实现叠加感
                    edgecolors='none')

# --- B. 绘制建筑长方体 (形象化) ---
def draw_building(ax, x, y, h, w, color='#a2d2ff'):
    v = np.array([[x-w, y-w, 0], [x+w, y-w, 0], [x+w, y+w, 0], [x-w, y+w, 0],
                  [x-w, y-w, h], [x+w, y-w, h], [x+w, y+w, h], [x-w, y+w, h]])
    faces = [[v[0], v[1], v[5], v[4]], [v[1], v[2], v[6], v[5]], 
             [v[2], v[3], v[7], v[6]], [v[3], v[0], v[4], v[7]], [v[4], v[5], v[6], v[7]]]
    poly = Poly3DCollection(faces, facecolors=color, edgecolors='black', alpha=0.5, linewidths=0.3)
    ax.add_collection3d(poly)

for b in buildings_data:
    draw_building(ax, b['x'], b['y'], b['z'], b['w']/2)
    ax.text(b['x'], b['y'], b['z']+10, b['name'], fontsize=7, ha='center')

# 标注释放源 (东方明珠)
ax.scatter(0, 0, 468, color='red', marker='*', s=300, label='Source (Pearl Tower)')

# --- C. 场景设置 ---
ax.set_xlim(-1000, 1000); ax.set_ylim(-1000, 1000); ax.set_zlim(0, 700)
ax.set_xlabel('West - East (m)'); ax.set_ylabel('South - North (m)'); ax.set_zlabel('Height (m)')
ax.set_title(f'3D Pollutant Dispersion Point Cloud (t={total_time}s)', fontsize=15)
ax.view_init(elev=25, azim=-60) # 设置观察视角

plt.colorbar(scatter, label='Concentration (Log Scale)', shrink=0.5)
plt.tight_layout()
plt.show()