import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata
import pandas as pd
import bianliang as vb

# 检查CUDA是否可用
device = vb.device
print(f"Using device: {device}")

# 定义相同的网络结构并移动到GPU
input_size = vb.input_size
output_size = vb.output_size

my_nn = vb.simplest_model

# 加载模型参数
my_nn.load_state_dict(torch.load('my_model.pth', map_location=device))
my_nn.eval()  # 设置为评估模式

# 加载scaler
scaler = joblib.load('scaler.save')

# 读取所有数据
data = pd.read_csv('truedata.csv')
z_true = data['results']  # 真实值
data = pd.read_csv('parameter_combinations.csv')
input_data = data[['参数1', '参数2', '参数3', '参数4', '参数5', '参数6']].values

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
x_var = data['参数1'] * data['参数2'] * data['参数3'] * data['参数6']
y_var = data['参数4'] * data['参数5'] * data['参数6']
z_var = predictions.numpy().flatten()  # 使用预测值

# 创建预测结果的DataFrame并保存
pred_df = pd.DataFrame({
    'SHV': x_var,
    'LLV': y_var,
    'D_predicted': z_var
})
pred_df.to_csv('prediction_results.csv', index=False)
# 创建网格
xi = np.linspace(x_var.min(), x_var.max(), 100)
yi = np.linspace(y_var.min(), y_var.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# 使用预测值进行插值
zi_pred = griddata((x_var, y_var), z_var, (xi, yi), method='cubic')

# 定义颜色映射范围
norm = plt.Normalize(10, 35)

#绘制预测的线框图
wire = ax.plot_wireframe(xi, yi, zi_pred, rstride=5, cstride=5, alpha=0.7,
                        label='Predicted Surface')
# wire = ax.plot_wireframe(x_var.values.reshape(25, 25), 
#                         y_var.values.reshape(25, 25), 
#                         z_var.reshape(25, 25),
#                         rstride=1, cstride=1, alpha=0.7,
#                         label='Predicted Surface')
#导入实验数据
data = pd.read_csv('truedata.csv')
x_var = data['参数1'] * data['参数2'] * data['参数3'] * data['参数6']
y_var = data['参数4'] * data['参数5'] * data['参数6']
z_true = data['truedata']  
#绘制实验数据点
scatter = ax.scatter(x_var, y_var, z_true,
                    color='red', s=50, label='True Values')

# 设置标签
ax.set_xlabel('复合参数1')
ax.set_ylabel('复合参数2')
ax.set_zlabel('results')

# 添加标题
plt.title('3D Wireframe Plot: Predicted Surface vs True Values')

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='trudata value')

# 添加图例
plt.legend()

# # 调整视角
# ax.view_init(elev=30, azim=45)

# 显示图形
plt.show()

# # 计算平均绝对误差
# mae = np.mean(np.abs(predictions.numpy() - z_true.values))
# print(f"\n平均绝对误差: {mae:.2f}")
