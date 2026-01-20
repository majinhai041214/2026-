import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ================= 1. 物理参数与网格初始化 =================
dx, dy, dz = 20, 20, 10
nx, ny, nz = 101, 101, 71
dt = 3.0
total_time = 7200
steps = int(total_time / dt)

x_vec = np.linspace(-1000, 1000, nx)
y_vec = np.linspace(-1000, 1000, ny)
z_vec = np.linspace(0, 700, nz)
X, Y = np.meshgrid(x_vec, y_vec, indexing='ij')

record_interval = 10
record_dt_min = dt * record_interval / 60.0

# ================= 2. 建筑场与小区 =================
buildings_cfg = [
    {"x": -420, "y": 80, "A": 2.5, "R": 150},
    {"x": 480, "y": 220, "A": 2.0, "R": 120},
    {"x": 550, "y": -120, "A": 2.2, "R": 180},
    {"x": -280, "y": -380, "A": 1.8, "R": 130},
    {"x": -550, "y": -250, "A": 2.1, "R": 140}
]

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

# ================= 3. 7 个风向（已合成江风）参数 =================
# theta: 度, K0: 已计算好的值, w: 权重
wind_cases = [
    {"name": "NNW", "theta": 20.07,  "K0": 6.00, "w": 0.1145},
    {"name": "N",   "theta": 33.21,  "K0": 8.02, "w": 0.2748},
    {"name": "NNE", "theta": 49.14,  "K0": 9.05, "w": 0.1221},
    {"name": "ENE", "theta": 77.10,  "K0": 10.0, "w": 0.0763},
    {"name": "ESE", "theta": 102.9,  "K0": 10.0, "w": 0.1221},
    {"name": "SE",  "theta": 116.31, "K0": 9.72, "w": 0.1832},
    {"name": "SSE", "theta": 131.8,  "K0": 9.32, "w": 0.1069},
]

# 权重归一化
wsum = sum(w["w"] for w in wind_cases)
for w in wind_cases:
    w["w"] /= wsum

alpha = 0.4
lambda_decay = 0.004
threshold = 1e-6

# ================= 4. 存储所有风向的结果 =================
all_history = {name: [] for name in neighborhoods}
time_axis = None

# ================= 5. 主循环：遍历 7 个风向 =================
for i, wind in enumerate(wind_cases, 1):
    print(f"\n[{i}/7] 开始计算风向 {wind['name']} ...")

    # 初始化浓度场
    C = np.zeros((nx, ny, nz))
    M = 1e15
    C[nx//2, ny//2, int(200/dz)] = M / (dx*dy*dz)

    # 扩散系数
    theta = np.deg2rad(wind["theta"])
    K0 = wind["K0"]

    Kx_2d = K0 * (np.cos(theta)**2 + alpha * np.sin(theta)**2) * Fbld
    Ky_2d = K0 * (np.sin(theta)**2 + alpha * np.cos(theta)**2) * Fbld
    Kz_vec = 2.0 * (1 - np.exp(-z_vec / 20.0))

    Kx = np.repeat(Kx_2d[:, :, None], nz, axis=2)
    Ky = np.repeat(Ky_2d[:, :, None], nz, axis=2)
    Kz = np.tile(Kz_vec, (nx, ny, 1))

    history = {name: [] for name in neighborhoods}
    local_time_axis = []

    for s in range(steps):
        C_old = C.copy()

        C[1:-1, 1:-1, 1:-1] += dt * (
            Kx[1:-1,1:-1,1:-1] *
            (C_old[2:,1:-1,1:-1] - 2*C_old[1:-1,1:-1,1:-1] + C_old[:-2,1:-1,1:-1]) / dx**2 +
            Ky[1:-1,1:-1,1:-1] *
            (C_old[1:-1,2:,1:-1] - 2*C_old[1:-1,1:-1,1:-1] + C_old[1:-1,:-2,1:-1]) / dy**2 +
            Kz[1:-1,1:-1,1:-1] *
            (C_old[1:-1,1:-1,2:] - 2*C_old[1:-1,1:-1,1:-1] + C_old[1:-1,1:-1,:-2]) / dz**2
            - lambda_decay * C_old[1:-1,1:-1,1:-1]
        )

        C[:, :, 0] = C[:, :, 1]
        C[:, :, -1] = C[:, :, -2]

        if s % record_interval == 0:
            t_min = s * dt / 60.0
            local_time_axis.append(t_min)

            for name, b in neighborhoods.items():
                mask = (X >= b["x"][0]) & (X <= b["x"][1]) & \
                       (Y >= b["y"][0]) & (Y <= b["y"][1])
                history[name].append(np.mean(C[:, :, 1][mask]))

        if s % 400 == 0:
            print(f"  风向 {wind['name']} 进度: {s/steps*100:.1f}%")

    if time_axis is None:
        time_axis = local_time_axis

    for name in neighborhoods:
        all_history[name].append((wind["w"], np.array(history[name])))

# ================= 6. 计算期望（加权）曲线 =================
combined = {}
for name, curves in all_history.items():
    combined[name] = sum(w * c for w, c in curves)

# ================= 7. 统计综合暴露时间 =================
exposure_minutes = {
    name: np.sum(curve > threshold) * record_dt_min
    for name, curve in combined.items()
}

# ================= 8. 绘制综合期望曲线 =================
plt.figure(figsize=(12, 7))
for name, curve in combined.items():
    plt.plot(time_axis, curve, label=name, linewidth=2)

plt.axhline(threshold, color='black', linestyle='--', label='Threshold')
plt.xlabel("Time after Release (minutes)")
plt.ylabel("Average Concentration ($\mu g/m^3$)")
plt.title("Expected Concentration Curve (Weighted over 7 Wind Directions)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ================= 9. 输出综合结果 =================
print("\n=== 综合暴露时间（分钟） ===")
for name, t in exposure_minutes.items():
    print(f"{name}: {t:.1f} min")
