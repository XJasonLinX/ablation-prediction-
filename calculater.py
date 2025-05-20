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
data = pd.read_csv('parameter_combinations.csv')
input_data = data[['SHVV', 'SHVW', 'SHVN', 'LLVV', 'LLVW', 'N']].values

# 使用保存的scaler进行归一化
input_feature = scaler.transform(input_data)

# 转换为tensor并移动到GPU进行预测
x = torch.tensor(input_feature, dtype=torch.float32).to(device)
with torch.no_grad():
    predictions = my_nn(x)
    predictions = predictions.cpu()  # 移回CPU用于绘图
  
z_var = predictions.numpy().flatten()  # 使用预测值

target_diameter = float(input('请输入目标直径: '))
error_margin = float(input('请输入允许误差: '))

# 找到所有符合条件的预测值的索引
valid_indices = np.where(np.abs(z_var - target_diameter) <= error_margin)[0]

if len(valid_indices) > 0:
    print("\n找到以下符合条件的参数组合：")
    print("\nSHVV, SHVW, SHVN, LLVV, LLVW, N, 预测直径")
    print("-" * 50)
    for idx in valid_indices:
        params = input_data[idx]
        predicted_d = z_var[idx]
        print(f"{params[0]}, {params[1]:.1f}, {params[2]}, "
              f"{params[3]}, {params[4]}, {params[5]}, "
              f"{predicted_d:.3f}")
    
    # 将结果保存到CSV文件
    results_df = pd.DataFrame(input_data[valid_indices], 
                            columns=['SHVV', 'SHVW', 'SHVN', 'LLVV', 'LLVW', 'N'])
    results_df['Predicted_Diameter'] = z_var[valid_indices]
    results_df.to_csv('valid_parameters.csv', index=False)
    print(f"\n找到 {len(valid_indices)} 组符合条件的参数")
else:
    print("\n未找到符合条件的参数组合")