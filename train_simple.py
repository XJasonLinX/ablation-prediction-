import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn import preprocessing
import warnings
import bianliang as vb

warnings.filterwarnings("ignore")

feature_df = pd.read_csv('yourdata.csv') 
labels_np = np.array(feature_df['results'])
feature_df = feature_df.drop(columns=['results'])

print("数据维度", feature_df.shape) 
feature_np = np.array(feature_df)

scaler = preprocessing.StandardScaler()
input_feature_scaled = scaler.fit_transform(feature_np)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

x_tensor = torch.tensor(input_feature_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(labels_np, dtype=torch.float32).unsqueeze(1).to(device) 

# --- 构建最简线性模型 ---
input_size = input_feature_scaled.shape[1]  
output_size = 1

simplest_model = vb.simplest_model

# --- 损失函数和优化器 ---
cost_fn = torch.nn.MSELoss(reduction='mean')
# optimizer_simple = optim.SGD(simplest_model.parameters(), lr=5e-5) 
optimizer_simple = optim.Adam(simplest_model.parameters(), lr=4e-4)


print("\n--- 开始训练 ---")
num_epochs_simple = 60000  

for epoch in range(num_epochs_simple):

    # 前向传播
    predictions = simplest_model(x_tensor)
    loss = cost_fn(predictions, y_tensor)

    # 反向传播和优化
    optimizer_simple.zero_grad()
    loss.backward()
    optimizer_simple.step()

    if epoch % 200 == 0:
        print(f"Simple Model - Epoch {epoch}, Loss: {loss.item():.6f}")
        with torch.no_grad():
            preds_np_flat = predictions.cpu().numpy().flatten()
            # 检查预测值是否趋同
            if len(np.unique(preds_np_flat)) < 5: # 如果唯一值很少，说明趋同
                print(f"  WARNING: Predictions are nearly constant: {np.unique(preds_np_flat)[:5]}")


# --- 训练后评估模型 ---
with torch.no_grad():
    final_predictions = simplest_model(x_tensor)
    final_loss = cost_fn(final_predictions, y_tensor)
    print(f"\n--- 模型最终结果 ---")
    print(f"Final Loss: {final_loss.item():.6f}")

    final_preds_np_flat = final_predictions.cpu().numpy().flatten()
    if len(np.unique(final_preds_np_flat)) < 5:
        print(f"FINAL WARNING: Predictions are nearly constant. Unique values: {np.unique(final_preds_np_flat)[:5]}")
        # 输出这个固定值是否接近 'results' 的均值
        mean_d = np.mean(labels_np)
        print(f"Mean of target 'results': {mean_d:.4f}. Constant prediction is around: {final_preds_np_flat[0]:.4f}")
    else:
        print(f"Predictions range: [{final_preds_np_flat.min():.4f}, {final_preds_np_flat.max():.4f}]")

    # 检查权重
    print("\nModel Weights and Biases:")
    for name, param in simplest_model.named_parameters():
        print(f"{name}: {param.data.cpu().numpy()}")
simplest_model.cpu()
torch.save(simplest_model.state_dict(), 'my_model3.pth')
print("模型已保存为 my_model3.pth")

# Save scaler
import joblib
joblib.dump(scaler, 'scaler3.save')
