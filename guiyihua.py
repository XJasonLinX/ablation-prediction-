import numpy as np
import pandas as pd
import warnings

# Data loading and preprocessing
feature = pd.read_csv('pre18282.csv')
labels = np.array(feature['D'])
feature = feature.drop(columns=['D'])
print("数据维度", feature.shape)
feature = np.array(feature)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
input_feature = scaler.fit_transform(feature)
print(input_feature)