import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata
import pandas as pd

# 加载模型参数
checkpoint = torch.load('my_model.pth')
weights1 = checkpoint['weights1']
biases1 = checkpoint['biases1']
weights2 = checkpoint['weights2']
biases2 = checkpoint['biases2']
weights3 = checkpoint['weights3']
biases3 = checkpoint['biases3']
# weights4 = checkpoint['weights4']
# biases4 = checkpoint['biases4']

# 加载scaler
scaler = joblib.load('scaler.save')

# 读取所有数据
data = pd.read_csv('pre18282.csv')
input_data = data[['SHVV', 'SHVW', 'SHVN', 'LLVV', 'LLVW', 'N']].values

# 使用保存的scaler进行归一化
input_feature = scaler.transform(input_data)

# 转换为tensor
x = torch.tensor(input_feature, dtype=float)

# 前向传播预测
hidden1 = x.mm(weights1) + biases1
hidden1 = torch.relu(hidden1)

hidden2 = hidden1.mm(weights2) + biases2
hidden2 = torch.relu(hidden2)

# hidden3 = hidden2.mm(weights3) + biases3
# hidden3 = torch.relu(hidden3)

predictions = hidden2.mm(weights3) + biases3

# 创建3D图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 计算复合变量
x_var = data['SHVV'] * data['SHVW'] * data['SHVN'] * data['N']
y_var = (data['LLVV'] * data['LLVW']) * data['N']
z_var = predictions.detach().numpy().flatten()  # 使用预测值
z_true = data['D']  # 真实值

# 创建网格
xi = np.linspace(x_var.min(), x_var.max(), 100)
yi = np.linspace(y_var.min(), y_var.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# 使用预测值进行插值
zi_pred = griddata((x_var, y_var), z_var, (xi, yi), method='cubic')

# 绘制预测的线框图
wire = ax.plot_wireframe(xi, yi, zi_pred, rstride=5, cstride=5, alpha=0.7, 
                        label='Predicted Surface')

# 绘制真实数据点
scatter = ax.scatter(x_var, y_var, z_true, 
                    color='red', s=50, label='True Values')

# 设置标签
ax.set_xlabel('SHVV*SHVW*SHVN*N')
ax.set_ylabel('LLVV*LLVW*N')
ax.set_zlabel('D')

# 添加标题
plt.title('3D Wireframe Plot: Predicted Surface vs True Values')

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(10, 35))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='D value')

# 添加图例
plt.legend()

# 调整视角
ax.view_init(elev=30, azim=45)

# 显示图形
plt.show()

# 打印预测结果与真实值的对比
print("\n预测结果与真实值对比：")
for i in range(len(predictions)):
    print(f"预测值: {predictions[i].item():.2f}, 真实值: {z_true.iloc[i]:.2f}")

# 计算平均绝对误差
mae = np.mean(np.abs(predictions.detach().numpy() - z_true.values))
print(f"\n平均绝对误差: {mae:.2f}")