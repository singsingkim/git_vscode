# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# import sklearn as sk
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# print(sk.__version__)   # 1.1.3

# # 1 데이터 
# datasets = load_diabetes()
# x = datasets['data']
# y = datasets.target
# print(x.shape, y.shape) # (150, 4) (150,)

# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)

# best_accuracy = 0
# best_n_components = 0

# for n in range(x.shape[1], 1, -1):
#     pca = PCA(n_components=n)
#     x_pca = pca.fit_transform(x_scaled)

#     x_train, x_test, y_train, y_test = train_test_split(
#         x_pca, y, train_size=0.8, random_state=123, shuffle=True
#     )

#     # 모델
#     model = RandomForestRegressor(random_state=123)

#     # 훈련
#     model.fit(x_train, y_train)

#     # 평가, 예측
#     accuracy = model.score(x_test, y_test)
#     print(f"Number of Components: {n}, Accuracy: {accuracy}")

#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_n_components = n

# print(f"\nBest Number of Components: {best_n_components}, Best Accuracy: {best_accuracy}")


# EVR = pca.explained_variance_ratio_ # 설명할수잇는변화율
# print(sum(EVR))






# 주성분분석이라 부름
# 스케일링 후 pca 후 트레인
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

print(sk.__version__)   # 1.1.3

# 1 데이터 
# datasets = load_diabetes()
datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)


scaler = StandardScaler()
x = scaler.fit_transform(x)


pca = PCA(n_components= 30)
x = pca.fit_transform(x)
print(x)
print(x.shape)  # (150, 4)

x_train, x_test, y_train, y_test = train_test_split(
    
    x, y, train_size=0.8, random_state=123, shuffle=True
)

# 2 모델
# model = RandomForestRegressor(random_state=123)
model = RandomForestClassifier(random_state=123)

# 3 훈련

model.fit(x_train, y_train)

# 4 평가, 예측
results = model.score(x_test, y_test)
print(x.shape)
print('model score : ', results)


evr = pca.explained_variance_ratio_     # 1.0 에 가까워야 좋다
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()






