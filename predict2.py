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

my_nn = vb.my_nn

# 加载模型参数
my_nn=torch.load('my_model4.pth', map_location=device)
my_nn.eval()  # 设置为评估模式

# 加载scaler
scaler = joblib.load('scaler3.save')

# 读取所有数据
data = pd.read_csv('pre18282.csv')
input_data = data[['SHVV', 'SHVW', 'SHVN', 'LLVV', 'LLVW', 'N']].values

# 使用保存的scaler进行归一化
input_feature = scaler.transform(input_data)


# 转换为tensor并移动到GPU进行预测
x = torch.tensor(input_feature, dtype=torch.float32).to(device)
with torch.no_grad():
    predictions = my_nn(x)
    predictions = predictions.cpu()  # 移回CPU用于绘图
  

# 创建3D图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 计算复合变量
x_var = data['SHVV'] * data['SHVW'] * data['SHVN'] * data['N']
y_var = data['LLVV'] * data['LLVW'] * data['N']
z_var = predictions.numpy().flatten()  # 使用预测值
z_true = data['D']  # 真实值

# 创建网格
xi = np.linspace(x_var.min(), x_var.max(), 100)
yi = np.linspace(y_var.min(), y_var.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# 使用预测值进行插值
zi_pred = griddata((x_var, y_var), z_var, (xi, yi), method='cubic')

# 定义颜色映射范围
norm = plt.Normalize(10, 35)

# 绘制预测的线框图
wire = ax.plot_wireframe(xi, yi, zi_pred, rstride=5, cstride=5, alpha=0.7,
                        label='Predicted Surface')

# 绘制真实数据点
scatter = ax.scatter(x_var, y_var, z_true,
                    color='red', s=50, label='True Values')

# 设置标签
ax.set_xlabel('SHV')
ax.set_ylabel('LLV')
ax.set_zlabel('D')

# 添加标题
plt.title('3D Wireframe Plot: Predicted Surface vs True Values (Model 2)')

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
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
mae = np.mean(np.abs(predictions.numpy() - z_true.values))
print(f"\n平均绝对误差: {mae:.2f}")