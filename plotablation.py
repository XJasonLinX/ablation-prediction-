import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata
import pandas as pd
import bianliang as vb

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义相同的网络结构并移动到GPU
input_size = 6
output_size = 1

my_nn = vb.simplest_model

# 加载模型参数
my_nn.load_state_dict(torch.load('my_model3.pth', map_location=device))
my_nn.eval()  # 设置为评估模式

# 加载scaler
scaler = joblib.load('scaler3.save')

# 读取所有数据
input_data = input('请输入SHVV，SHVW，SHVN，LLVV，LLVW，N:')
# 将输入字符串转换为数值列表
input_values = [float(x.strip()) for x in input_data.split(',')]
# 转换为numpy数组并reshape为2D数组(因为scaler需要2D输入)
input_data = np.array(input_values).reshape(1, -1)
# 使用保存的scaler进行归一化
input_feature = scaler.transform(input_data)

# 转换为tensor并移动到GPU进行预测
x = torch.tensor(input_feature, dtype=torch.float32).to(device)
with torch.no_grad():
    predictions = my_nn(x)
    predictions = predictions.cpu()  # 移回CPU用于绘图
print(predictions)
d=float(predictions)
print(d)

d_param = d  # 您可以更改此值

# 绘图范围和点数
x_min, x_max = -20, 20  # x 的范围受 (1 - x^2/19^2) 项的限制
num_points_x = 40  # x轴上的点数
num_points_theta = 30  # theta角上的点数 (用于构建圆形截面)
# --- 结束用户可配置参数 ---

# 创建 x 和 theta 的值域
# 我们限制x在[-19, 19]之间，因为超出这个范围 (1 - x^2/19^2) 会变为负数，其3/4次方会产生复数或错误。
# 为了避免在边界处出现除零或无效值错误（尽管(0)^(3/4)是0），我们可以稍微缩小范围或小心处理。
# 这里我们使用 linspace，它会包含端点。
x_vals = np.linspace(x_min, x_max, num_points_x)
theta_vals = np.linspace(-25, 25)

# 创建网格
X, THETA = np.meshgrid(x_vals, theta_vals)

# 方程的右边部分，我们称之为 R_squared_func(x, d)
# z^2 + y^2 = R_squared_func
# R_squared_func = (2/3) * (sqrt(4*(9^2*x^2) + d^4) - x^2 - 9^2) * (1 - x^2/19^2)^(3/4)

# 计算 (1 - X^2/19^2)^(3/4) 项
# 我们需要确保基数 (1 - X^2/19^2) 是非负的。
# 由于X的范围已限定在[-19, 19]，所以基数是 >= 0。
term_factor = (1 - (X**2) / (19**2))
# 处理由于浮点精度问题可能导致的小负数
term_factor[term_factor < 0] = 0
term_power = term_factor**(3/4)

# 计算 sqrt(4*(9^2*X^2) + d_param^4) - X^2 - 9^2 项
term_sqrt = np.sqrt(4 * (81 * (X**2)) + d_param**4)
term_sub = term_sqrt - (X**2) - (9**2)
R_squared = (2/3) * term_sub * term_power

# 由于 R_squared = y^2 + z^2，它必须是非负的。
# 如果计算结果为负，则在这些(x, theta)点上没有实数解。
# 我们将这些点的半径设为0 (或 NaN 以不绘制它们)。
R = np.zeros_like(R_squared)
positive_R_squared_indices = R_squared >= 0
R[positive_R_squared_indices] = np.sqrt(R_squared[positive_R_squared_indices])

Y = R * np.cos(THETA)
Z = R * np.sin(THETA)




# 创建3D图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


ax.set_box_aspect([1, 1, 0.9])  # 等比例 XYZ 轴

# 绘制曲面
# 使用 X, Y, Z。注意 X 已经是一个网格了。
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1, alpha=0.8)

# 设置坐标轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_zlim(-25, 25)
ax.set_ylim(-25, 25)

# 设置标题
# ax.set_title(f'3D Plot of $z^2+y^2 = \\frac{2}{3}(\\sqrt{4(9^2*x^2)+d^4}-x^2-9^2)(1-\\frac{x^2}{19^2})^{{3/4}}$\nwith d={d_param}')

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5, label='Radius (approx.)')

# 确保所有元素都可见
plt.tight_layout()
plt.show()

print(f"Parameter 'd' was set to: {d_param}")
print(f"X range: [{x_min}, {x_max}]")
if np.any(R_squared < 0):
    print("Note: Some regions resulted in R_squared < 0 and were not plotted or R set to 0.")
else:
    print("All calculated R_squared values were non-negative.")

