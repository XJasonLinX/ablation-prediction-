import pandas as pd

# 定义参数及其可能的值
parameters = {
    '参数1': [1, 2, 3, 4, 5],
    '参数2': [1, 2, 3, 4, 5],
    '参数3': [1, 2, 3, 4, 5],
    '参数4': [1, 2, 3, 4, 5],
    '参数5': [1, 2, 3, 4, 5],
    '参数6': [1, 2, 3, 4, 5]
}

# 生成所有组合的索引
index = pd.MultiIndex.from_product(parameters.values(), names=parameters.keys())

# 转换为 DataFrame
df = pd.DataFrame(index=index).reset_index()

# 保存为 CSV
df.to_csv('parameter_combinations.csv', index=False)

print("CSV 文件已生成: parameter_combinations.csv")
