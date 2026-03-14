import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 参数设置
# =========================================================
lam = 500e-9      # 入射光波长
d = 2e-3          # 缝缝宽度/双缝间距
D = 1             # 双缝与屏幕水平间距

# ym = 5*lam*D/d，原MATLAB里写的是 5*lam*d/d，其实通常应为 5*lam*D/d
ym = 5 * lam * D / d

xs = ym
n = 101           # 取点个数
ys = np.linspace(-ym, ym, n)

# =========================================================
# 计算干涉强度
# =========================================================
B = np.zeros(n)

for i in range(n):
    r1 = np.sqrt((ys[i] - d/2)**2 + D**2)
    r2 = np.sqrt((ys[i] + d/2)**2 + D**2)

    phi = 2 * np.pi * (r2 - r1) / lam

    # 对应 MATLAB: B(i,:) = sum(4*cos(phi/2).^2)
    # 因为这里 phi 是单个标量，sum 实际没有必要
    B[i] = 4 * np.cos(phi / 2)**2

# =========================================================
# 灰度级处理
# =========================================================
N = 255
Br = B * N   # 相干光强

# 为了显示成二维条纹图，把一维强度沿 x 方向复制
Br2D = np.tile(Br.reshape(-1, 1), (1, n))

# =========================================================
# 作图
# =========================================================
plt.figure(figsize=(10, 4))

# 左图：二维干涉条纹
plt.subplot(1, 2, 1)
plt.imshow(
    Br2D,
    cmap='gray',
    origin='lower',
    extent=[-xs, xs, -ym, ym],
    aspect='auto'
)
plt.title("2D Interference Pattern")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

# 右图：一维光强分布
plt.subplot(1, 2, 2)
plt.plot(B, ys, 'b', linewidth=2)
plt.title("Intensity Distribution")
plt.xlabel("Intensity")
plt.ylabel("y (m)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()