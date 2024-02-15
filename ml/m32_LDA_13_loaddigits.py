import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
from sklearn.pipeline import Pipeline



# 1 데이터
datasets = load_digits()

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
    model = RandomForestClassifier(random_state=123)

    # 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    accuracy = model.score(x_test, y_test)
    print(f"Number of Components: {n}, Accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_components = n

print(f"\nBest Number of Components: {best_n_components}, Best Accuracy: {best_accuracy}")


# Number of Components: 64, Accuracy: 0.9611111111111111
# Number of Components: 63, Accuracy: 0.9666666666666667
# Number of Components: 62, Accuracy: 0.9611111111111111
# Number of Components: 61, Accuracy: 0.9527777777777777
# Number of Components: 60, Accuracy: 0.9611111111111111
# Number of Components: 59, Accuracy: 0.9555555555555556
# Number of Components: 58, Accuracy: 0.9666666666666667
# Number of Components: 57, Accuracy: 0.9583333333333334
# Number of Components: 56, Accuracy: 0.9666666666666667
# Number of Components: 55, Accuracy: 0.9555555555555556
# Number of Components: 54, Accuracy: 0.9638888888888889
# Number of Components: 53, Accuracy: 0.9722222222222222
# Number of Components: 52, Accuracy: 0.9555555555555556
# Number of Components: 51, Accuracy: 0.9722222222222222
# Number of Components: 50, Accuracy: 0.9638888888888889
# Number of Components: 49, Accuracy: 0.9638888888888889
# Number of Components: 48, Accuracy: 0.9611111111111111
# Number of Components: 47, Accuracy: 0.9611111111111111
# Number of Components: 46, Accuracy: 0.9611111111111111
# Number of Components: 45, Accuracy: 0.9722222222222222
# Number of Components: 44, Accuracy: 0.9694444444444444
# Number of Components: 43, Accuracy: 0.9666666666666667
# Number of Components: 42, Accuracy: 0.9694444444444444
# Number of Components: 41, Accuracy: 0.9666666666666667
# Number of Components: 40, Accuracy: 0.9555555555555556
# Number of Components: 39, Accuracy: 0.9611111111111111
# Number of Components: 38, Accuracy: 0.9694444444444444
# Number of Components: 37, Accuracy: 0.9583333333333334
# Number of Components: 36, Accuracy: 0.9694444444444444
# Number of Components: 35, Accuracy: 0.9611111111111111
# Number of Components: 34, Accuracy: 0.9666666666666667
# Number of Components: 33, Accuracy: 0.9611111111111111
# Number of Components: 32, Accuracy: 0.9694444444444444
# Number of Components: 31, Accuracy: 0.9638888888888889
# Number of Components: 30, Accuracy: 0.95
# Number of Components: 29, Accuracy: 0.9666666666666667
# Number of Components: 28, Accuracy: 0.9555555555555556
# Number of Components: 27, Accuracy: 0.9555555555555556
# Number of Components: 26, Accuracy: 0.9583333333333334
# Number of Components: 25, Accuracy: 0.9583333333333334
# Number of Components: 24, Accuracy: 0.9583333333333334
# Number of Components: 23, Accuracy: 0.9638888888888889
# Number of Components: 22, Accuracy: 0.9611111111111111
# Number of Components: 21, Accuracy: 0.9583333333333334
# Number of Components: 20, Accuracy: 0.9555555555555556
# Number of Components: 19, Accuracy: 0.9611111111111111
# Number of Components: 18, Accuracy: 0.9555555555555556
# Number of Components: 17, Accuracy: 0.9555555555555556
# Number of Components: 16, Accuracy: 0.95
# Number of Components: 15, Accuracy: 0.9555555555555556
# Number of Components: 14, Accuracy: 0.9472222222222222
# Number of Components: 13, Accuracy: 0.9305555555555556
# Number of Components: 12, Accuracy: 0.9277777777777778
# Number of Components: 11, Accuracy: 0.9333333333333333
# Number of Components: 10, Accuracy: 0.9194444444444444
# Number of Components: 9, Accuracy: 0.9361111111111111
# Number of Components: 8, Accuracy: 0.9222222222222223
# Number of Components: 7, Accuracy: 0.9305555555555556
# Number of Components: 6, Accuracy: 0.8805555555555555
# Number of Components: 5, Accuracy: 0.8638888888888889
# Number of Components: 4, Accuracy: 0.8472222222222222
# Number of Components: 3, Accuracy: 0.7833333333333333
# Number of Components: 2, Accuracy: 0.5694444444444444

# Best Number of Components: 53, Best Accuracy: 0.9722222222222222
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

