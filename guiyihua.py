import numpy as np
import pandas as pd
import warnings

# Data loading and preprocessing
feature = pd.read_csv('truedata.csv')
labels = np.array(feature['results'])
feature = feature.drop(columns=['results'])
print("数据维度", feature.shape)
feature = np.array(feature)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
input_feature = scaler.fit_transform(feature)
print(input_feature)
