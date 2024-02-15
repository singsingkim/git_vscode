import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# 1. Data
# x,y = load_boston(return_X_y=True)
datasets = load_boston()
x = datasets.data
y = datasets.target
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

best_accuracy = 0
best_n_components = 0

for n in range(x.shape[1], 1, -1):
    pca = PCA(n_components=n)
    x_pca = pca.fit_transform(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pca, y, train_size=0.8, random_state=123, shuffle=True, # stratify=y
    )

    # 모델
    model = RandomForestRegressor(random_state=123)

    # 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    accuracy = model.score(x_test, y_test)
    print(f"Number of Components: {n}, Accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_components = n

print(f"\nBest Number of Components: {best_n_components}, Best Accuracy: {best_accuracy}")


# Number of Components: 13, Accuracy: 0.784378968227525
# Number of Components: 12, Accuracy: 0.7960331218676766
# Number of Components: 11, Accuracy: 0.7885063405193504
# Number of Components: 10, Accuracy: 0.7860169010300857
# Number of Components: 9, Accuracy: 0.7569931583855996
# Number of Components: 8, Accuracy: 0.7616586453446792
# Number of Components: 7, Accuracy: 0.7152959018651392
# Number of Components: 6, Accuracy: 0.695356401370938
# Number of Components: 5, Accuracy: 0.6726222662650785
# Number of Components: 4, Accuracy: 0.6285574489453087
# Number of Components: 3, Accuracy: 0.6108543421894581
# Number of Components: 2, Accuracy: 0.5485490888433178

# Best Number of Components: 12, Best Accuracy: 0.7960331218676766
# PS C:\Study> 


evr = pca.explained_variance_ratio_     # 1.0 에 가까워야 좋다
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()

