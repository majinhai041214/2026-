import numpy as np
import matplotlib.pyplot as plt

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

alpha = 0.4
lambda_decay = 0.004
threshold = 1e-6

# ================= 2. 上海：物理场构建（建筑）与区域定义 =================
buildings_cfg = [
    {"name": "Ocean Aquarium",     "x": 210, "y": 50,   "A": 1.2, "R": 80},
    {"name": "Convention Center",  "x": -100,"y": 85,   "A": 1.5, "R": 90},
    {"name": "Super Brand Mall",   "x": -95, "y": -385, "A": 1.8, "R": 100},
    {"name": "IFC",               "x": 195, "y": -410, "A": 3.0, "R": 150},
    {"name": "Bank of China",     "x": 335, "y": -160, "A": 2.5, "R": 120},
    {"name": "Jin Mao Tower",     "x": 610, "y": -530, "A": 3.8, "R": 180},
    {"name": "SWFC",              "x": 780, "y": -540, "A": 4.0, "R": 180},
    {"name": "Shanghai Tower",    "x": 650, "y": -720, "A": 4.5, "R": 200},
]

neighborhoods = {
    "IFC": {"x": [150, 300], "y": [-500, -300]},
    "Jin Mao Tower": {"x": [500, 600], "y": [-500, -400]},
    "Shanghai Tower": {"x": [600, 800], "y": [-800, -600]}
}

Fbld = np.ones((nx, ny))
for b in buildings_cfg:
    Fbld += b["A"] * np.exp(-((X - b["x"])**2 + (Y - b["y"])**2) / b["R"]**2)

# ================= 3. 上海：7 个风向（复合角度 + 复合风速）与权重 =================
# 你要求：风速换算为 K0：每 40 km/h -> 7 K0
# 所以：K0 = wind_speed_kmh * 7 / 40
wind_cases = [
    {"name": "NW",  "theta": 315.0, "wind_speed": 30.4, "w": 0.0806},
    {"name": "N",   "theta": 11.7,  "wind_speed": 33.9, "w": 0.1552},
    {"name": "NE",  "theta": 58.5,  "wind_speed": 40.9, "w": 0.1264},
    {"name": "E",   "theta": 98.2,  "wind_speed": 47.3, "w": 0.1327},
    {"name": "ESE", "theta": 116.3, "wind_speed": 49.0, "w": 0.1475},
    {"name": "SE",  "theta": 135.0, "wind_speed": 49.6, "w": 0.1963},
    {"name": "SSE", "theta": 156.3, "wind_speed": 49.0, "w": 0.1613},
]

# 权重归一化（与广州版一致的做法）
wsum = sum(w["w"] for w in wind_cases)
for w in wind_cases:
    w["w"] /= wsum

# 把 wind_speed 换算为 K0（只做你要求的换算，不动任何其他算法）
for w in wind_cases:
    w["K0"] = w["wind_speed"] * 3.0 / 40.0

# ================= 4. 污染源高度：上海 600m（你指定） =================
source_height_m = 200
source_k = int(source_height_m / dz)

# ================= 5. 存储所有风向的结果 =================
all_history = {name: [] for name in neighborhoods}
time_axis = None

print("开始上海风向加权综合模拟（完全沿用广州算法，仅平移参数）...")

# ================= 6. 主循环：遍历 7 个风向 =================
for i, wind in enumerate(wind_cases, 1):
    print(f"\n[{i}/7] 开始计算风向 {wind['name']} ...")

    # 初始化浓度场
    C = np.zeros((nx, ny, nz))
    M = 1e15
    C[nx//2, ny//2, source_k] = M / (dx * dy * dz)

    # 完全沿用广州：theta 直接 deg2rad
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
            Kx[1:-1, 1:-1, 1:-1] *
            (C_old[2:, 1:-1, 1:-1] - 2*C_old[1:-1, 1:-1, 1:-1] + C_old[:-2, 1:-1, 1:-1]) / dx**2 +
            Ky[1:-1, 1:-1, 1:-1] *
            (C_old[1:-1, 2:, 1:-1] - 2*C_old[1:-1, 1:-1, 1:-1] + C_old[1:-1, :-2, 1:-1]) / dy**2 +
            Kz[1:-1, 1:-1, 1:-1] *
            (C_old[1:-1, 1:-1, 2:] - 2*C_old[1:-1, 1:-1, 1:-1] + C_old[1:-1, 1:-1, :-2]) / dz**2
            - lambda_decay * C_old[1:-1, 1:-1, 1:-1]
        )

        # z 边界：与广州一致
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

# ================= 7. 计算期望（加权）曲线 =================
combined = {}
for name, curves in all_history.items():
    combined[name] = sum(w * c for w, c in curves)

# ================= 8. 统计综合暴露时间 =================
exposure_minutes = {
    name: np.sum(curve > threshold) * record_dt_min
    for name, curve in combined.items()
}

# ================= 9. 绘制综合期望曲线（沿用广州版，不做额外“算法修正”） =================
plt.figure(figsize=(12, 7))
for name, curve in combined.items():
    plt.plot(time_axis, curve, label=name, linewidth=2)

plt.axhline(threshold, color='black', linestyle='--', label='Threshold')
plt.xlabel("Time after Release (minutes)")
plt.ylabel(r"Average Concentration ($\mu g/m^3$)")
plt.title("Shanghai Expected Concentration Curve (Weighted over 7 Wind Directions)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ================= 10. 输出综合结果 =================
print("\n=== 上海：综合暴露时间（分钟） ===")
for name, t in exposure_minutes.items():
    print(f"{name}: {t:.1f} min")