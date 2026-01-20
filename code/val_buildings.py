import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ================= 1. 核心建筑数据 (基于表) =================
buildings_info = [
    {"name": "Ocean Aquarium", "x": 210, "y": 50, "z": 30, "w": 100, "color": "#58DCFF"},
    {"name": "Convention Center", "x": -100, "y": 85, "z": 55, "w": 120, "color": "#92C6FD"},
    {"name": "Super Brand Mall", "x": -95, "y": -385, "z": 50, "w": 150, "color": "#3C4856"},
    {"name": "IFC", "x": 195, "y": -410, "z": 255, "w": 100, "color": "#A0ACBD"},
    {"name": "Bank of China", "x": 335, "y": -160, "z": 226, "w": 80, "color": "#92C6FD"},
    {"name": "Jin Mao Tower", "x": 610, "y": -530, "z": 420, "w": 90, "color": "#009AFA"},
    {"name": "SWFC", "x": 780, "y": -540, "z": 492, "w": 80, "color": "#F2FAFF"},
    {"name": "Shanghai Tower", "x": 650, "y": -720, "z": 632, "w": 100, "color": "#E6F4F1"}
]

def draw_vivid_building(ax, x, y, height, width, base_color):
    """绘制带楼层细线的形象化长方体建筑"""
    w = width / 2
    z = height
    # 定义8个顶点
    v = np.array([
        [x-w, y-w, 0], [x+w, y-w, 0], [x+w, y+w, 0], [x-w, y+w, 0],
        [x-w, y-w, z], [x+w, y-w, z], [x+w, y+w, z], [x-w, y+w, z]
    ])
    
    # 定义6个面
    faces = [
        [v[0], v[1], v[5], v[4]], # 前
        [v[1], v[2], v[6], v[5]], # 右
        [v[2], v[3], v[7], v[6]], # 后
        [v[3], v[0], v[4], v[7]], # 左
        [v[4], v[5], v[6], v[7]]  # 顶
    ]
    
    # 绘制侧面和顶部，添加阴影色差
    face_colors = [base_color, base_color, base_color, base_color, '#ffffff']
    poly = Poly3DCollection(faces, facecolors=face_colors, edgecolors='black', linewidths=0.3, alpha=0.8)
    ax.add_collection3d(poly)
    
    # 添加“楼层线”增加形象感 (每隔20米画一圈)
    floor_step = 20
    for h in range(floor_step, int(z), floor_step):
        ax.plot([x-w, x+w, x+w, x-w, x-w], [y-w, y-w, y+w, y+w, y-w], [h, h, h, h, h], 
                color='black', linewidth=0.1, alpha=0.5)

# ================= 2. 绘图初始化 =================
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制东方明珠 (特殊处理：细长塔身+球体抽象)
tower_x, tower_y, tower_z = 0, 0, 468
ax.plot([0, 0], [0, 0], [0, tower_z], color='red', linewidth=4, alpha=0.8)
ax.scatter([0], [0], [tower_z], color='red', s=200, marker='*', label='Oriental Pearl (Source)')
ax.scatter([0, 0], [0, 0], [150, 260], color='darkred', s=100) # 模拟球体位置

# 绘制其他8个建筑
for b in buildings_info:
    draw_vivid_building(ax, b['x'], b['y'], b['z'], b['w'], b['color'])
    ax.text(b['x'], b['y'], b['z']+15, b['name'], fontsize=8, ha='center', fontweight='bold')

# ================= 3. 场景美化 =================
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(0, 700)

# 设置白色画布与轴背景
fig.patch.set_facecolor('white')
ax.set_facecolor('#FFFFFF')
ax.grid(True, linestyle='--', alpha=0.3, color='#7CB342')

ax.set_xlabel('West - East (m)')
ax.set_ylabel('South - North (m)')
ax.set_zlabel('Height (m)')
ax.set_title('Lujiazui 3D Spatial Model (Yellow-Blue-Green Theme)', fontsize=15)

# 调整视角以获得最佳形象感
ax.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.show()