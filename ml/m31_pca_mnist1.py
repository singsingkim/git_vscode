# # #################### 실습 ######################
# # # pca 를 통해 0.95 이상인 n_components 는 몇개?
# # # 0.95 이상
# # # 0.99 이상
# # # 0.999 이상
# # # 1 일때 몇개?

# # 힌트 np.argmax
# ###############################################

from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# MNIST 데이터셋 로드
(x_train, _), (x_test, _) = mnist.load_data()

# 합치기
x = np.concatenate([x_train, x_test], axis=0)   # (70000, 28, 28)

x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
print(x.shape)  # (70000, 784)

pca = PCA(n_components=784)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_

cumsum = np.cumsum(evr)
print(cumsum)

print(np.argmax(cumsum >= 0.95) + 1)    # 154
print(np.argmax(cumsum >= 0.99) + 1)    # 331
print(np.argmax(cumsum >= 0.999) + 1)   # 486
print(np.argmax(cumsum >= 1.0) + 1)     # 713

