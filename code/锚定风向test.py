import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ================= 1. 物理参数与网格初始化 =================
color_list = ["#FFCF36", "#9BC34A", "#40AB6C", "#008C7F", "#006A78", "#2F4858"]
cmap_custom = LinearSegmentedColormap.from_list("custom", color_list[::-1])

dx, dy, dz = 20, 20, 10
nx, ny, nz = 101, 101, 71
dt = 3.0    # 30s  
total_time = 7200  # 延长模拟至 1 小时，观察浓度上升全过程
steps = int(total_time / dt)    #1200 步

x_vec = np.linspace(-1000, 1000, nx)
y_vec = np.linspace(-1000, 1000, ny)
z_vec = np.linspace(0, 700, nz)
X, Y = np.meshgrid(x_vec, y_vec, indexing='ij')

# ================= 2. 物理场构建与小区定义 =================
buildings_cfg = [
    {"name": "Media Port", "x": -420, "y": 80, "A": 2.5, "R": 150},
    {"name": "PR North", "x": 480, "y": 220, "A": 2.0, "R": 120},
    {"name": "PR East", "x": 550, "y": -120, "A": 2.2, "R": 180},
    {"name": "Boyawan", "x": -280, "y": -380, "A": 1.8, "R": 130},
    {"name": "Dijing Huayuan", "x": -550, "y": -250, "A": 2.1, "R": 140}
]

# 定义各小区计算边界
neighborhoods = {
    "Media Port": {"x": [-550, -300], "y": [0, 200]},
    "PR North": {"x": [400, 600], "y": [150, 300]},
    "PR East": {"x": [450, 750], "y": [-250, 50]},
    "Boyawan": {"x": [-400, -150], "y": [-500, -250]},
    "Dijing Huayuan": {"x": [-700, -450], "y": [-350, -150]}
}

Fbld = np.ones((nx, ny))
for b in buildings_cfg:
    Fbld += b['A'] * np.exp(-((X - b['x'])**2 + (Y - b['y'])**2) / b['R']**2)

# 各向异性扩散与垂直梯度（K0=3）



K0, theta, alpha = 9.32, np.deg2rad(131.8), 0.4




Kx_2d = K0 * (np.cos(theta)**2 + alpha * np.sin(theta)**2) * Fbld
Ky_2d = K0 * (np.sin(theta)**2 + alpha * np.cos(theta)**2) * Fbld
Kz_vec = 2.0 * (1 - np.exp(-z_vec / 20.0))

Kx = np.repeat(Kx_2d[:, :, np.newaxis], nz, axis=2)
Ky = np.repeat(Ky_2d[:, :, np.newaxis], nz, axis=2)
Kz = np.tile(Kz_vec, (nx, ny, 1))

# ================= 3. 时间序列模拟循环 =================
C = np.zeros((nx, ny, nz))
M = 1e15  # 调整为更合理的释放量以方便观察
C[nx//2, ny//2, int(600/dz)] = M / (dx*dy*dz) 

# 用于存储结果的字典
history = {name: [] for name in neighborhoods.keys()}
time_axis = []

# 统计小区区域平均暴露浓度超过 1e-6 的时间
exposure_times = {name: 0 for name in neighborhoods.keys()}

# 设定浓度阈值与评级标准
thresholds = [1e-6, 1e-5, 1e-4, 1e-3]  
labels = ["moderate", "medium", "severe", "very severe"]

print(f"开始时间序列模拟...")

for s in range(steps):
    C_old = C.copy()
    
    # --- 核心模型：三维扩散 + 线性衰减 (-0.01*C) ---
    term_x = (Kx[1:-1, 1:-1, 1:-1] * (C_old[2:, 1:-1, 1:-1] - 2*C_old[1:-1, 1:-1, 1:-1] + C_old[:-2, 1:-1, 1:-1])) / dx**2
    term_y = (Ky[1:-1, 1:-1, 1:-1] * (C_old[1:-1, 2:, 1:-1] - 2*C_old[1:-1, 1:-1, 1:-1] + C_old[1:-1, :-2, 1:-1])) / dy**2
    term_z = (Kz[1:-1, 1:-1, 1:-1] * (C_old[1:-1, 1:-1, 2:] - 2*C_old[1:-1, 1:-1, 1:-1] + C_old[1:-1, 1:-1, :-2])) / dz**2

    # 2. 定义衰减系数 (λ = 0.004)
    lambda_decay = 0.004

    # 3. 更新浓度 C
    C[1:-1, 1:-1, 1:-1] += dt * (term_x + term_y + term_z - lambda_decay * C_old[1:-1, 1:-1, 1:-1])

    # 边界条件：地面反射
    C[:, :, 0] = C[:, :, 1]
    C[:, :, -1] = C[:, :, -2]

    # 每隔 10 个时间步记录一次小区平均浓度 (减少计算开销)
    if s % 10 == 0:
        current_time = s * dt
        time_axis.append(current_time / 60.0)  # 转为分钟

        for name, bounds in neighborhoods.items():
            # 空间索引掩码提取
            mask = (X >= bounds["x"][0]) & (X <= bounds["x"][1]) & \
                   (Y >= bounds["y"][0]) & (Y <= bounds["y"][1])
            # 计算 10m 高度 (index 1) 的平均浓度
            avg_c = np.mean(C[:, :, 1][mask])
            history[name].append(avg_c)

            # 统计超过 1e-6 的时间
            if avg_c > 1e-6:
                exposure_times[name] += 1

    if s % 200 == 0:
        print(f"进度: {s/steps*100:.1f}%")

# ================= 4. 绘制时间序列结果图 =================
plt.figure(figsize=(12, 7))
for name in history:
    plt.plot(time_axis, history[name], label=name, linewidth=2)

# 绘制临界值虚线
plt.axhline(1e-6, color='black', linestyle='--', label='Threshold = 1e-6')

# 设置图像显示
plt.title("Average Concentration at 10m Height over Time", fontsize=14)
plt.xlabel("Time after Release (minutes)", fontsize=12)
plt.ylabel("Average Concentration ($\mu g/m^3$)", fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.ylim(0, max(max(history.values())) * 1.1)  # 纵轴范围适应数据最大值，留出一点空间
plt.legend()

# 自动调整布局
plt.tight_layout()
plt.show()

# ================= 5. 数据导出 =================
df_history = pd.DataFrame(history)
df_history.insert(0, 'Time_min', time_axis)
df_history.to_csv('neighborhood_time_series.csv', index=False)
print("小区时间序列数据已保存。")

# ================= 6. 评级计算 =================
for name, exposure_time in exposure_times.items():
    if exposure_time <= 10:
        rating = "moderate"
    elif exposure_time <= 20:
        rating = "medium"
    elif exposure_time <= 30:
        rating = "severe"
    else:
        rating = "very severe"
    # 输出小区名称、预警级别和暴露时间
    print(f"{name}: {rating}, Exposure time = {exposure_time} minutes")
