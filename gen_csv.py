import pandas as pd

# 定义参数及其可能的值
parameters = {
    'SHVV': [3000, 3500, 4000, 4500, 5000],
    'SHVW': [0.1, 0.2, 0.3, 0.4, 0.5],
    'SHVN': [0, 2, 4, 6, 8],
    'LLVV': [500, 800, 1000, 1200, 1500],
    'LLVW': [10, 20, 30, 40, 50],
    'N': [20, 40, 60, 80, 100]
}

# 生成所有组合的索引
index = pd.MultiIndex.from_product(parameters.values(), names=parameters.keys())

# 转换为 DataFrame
df = pd.DataFrame(index=index).reset_index()

# 保存为 CSV
df.to_csv('parameter_combinations.csv', index=False)

print("CSV 文件已生成: parameter_combinations.csv")
