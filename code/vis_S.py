import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ================= 1. 物理参数与函数定义 =================
def get_gaussian_pulse(X, Y, Z, t, xs=0, ys=0, H=600, sigma0=20, v_diff=5):
    # 随时间动态增长的 sigma (弥散参数)
    sigma = sigma0 + v_diff * t
    
    # 三维高斯公式
    prefactor = 1 / ((2 * np.pi)**(1.5) * sigma**3)
    exponent = -( ((X - xs)**2 + (Y - ys)**2 + (Z - H)**2) / (2 * sigma**2) )
    return prefactor * np.exp(exponent)

# ================= 2. 初始空间设置 =================
res = 40  # 交互模式下分辨率设为40以保证滑动流畅
x = np.linspace(-400, 400, res)
y = np.linspace(-400, 400, res)
z = np.linspace(300, 900, res)
X, Y, Z = np.meshgrid(x, y, z)

# ================= 3. 绘图初始化 =================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25) # 为滑块留出空间

# 初始时刻 t=0
t_init = 0
G = get_gaussian_pulse(X, Y, Z, t_init)
mask = G > (G.max() * 0.1)

# 绘制初始散点
scat = ax.scatter(X[mask], Y[mask], Z[mask], c=G[mask], cmap='YlOrRd', s=10, alpha=0.3)

# 设置轴标签与范围
ax.set_xlim(-400, 400); ax.set_ylim(-400, 400); ax.set_zlim(300, 900)
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Height Z (m)')
title = ax.set_title(f'Pulse Diffusion at t = {t_init}s', fontsize=14)

# ================= 4. 添加滑块控制 =================
ax_time = plt.axes([0.2, 0.1, 0.6, 0.03]) # 滑块位置
s_time = Slider(ax_time, 'Time (s)', 0, 100, valinit=t_init, valstep=1)

def update(val):
    t = s_time.val
    # 清除旧点并重新计算
    ax.cla() 
    ax.set_xlim(-400, 400); ax.set_ylim(-400, 400); ax.set_zlim(300, 900)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Height Z (m)')
    
    # 计算新时刻的浓度
    G_new = get_gaussian_pulse(X, Y, Z, t)
    mask_new = G_new > (G_new.max() * 0.1)
    
    # 重新绘图
    ax.scatter(X[mask_new], Y[mask_new], Z[mask_new], c=G_new[mask_new], 
               cmap='YlOrRd', s=10, alpha=0.3)
    ax.scatter(0, 0, 600, color='blue', marker='*', s=150) # 释放源中心
    ax.set_title(f'Pulse Diffusion at t = {val:.0f}s', fontsize=14)
    fig.canvas.draw_idle()

s_time.on_changed(update)
plt.show()