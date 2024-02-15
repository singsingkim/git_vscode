# 주성분분석이라 부름
# 트레인 후에 스케일링 후 pca
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
print(sk.__version__)   # 1.1.3

# 1 데이터 
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)


# scaler = StandardScaler()
# x = scaler.fit_transform(x)


# pca = PCA(n_components=4)
# x = pca.fit_transform(x)
# print(x)
# print(x.shape)  # (150, 4)

x_train, x_test, y_train, y_test = train_test_split(
    
    x, y, train_size=0.8, random_state=123, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


print(x_train)
print(x_train.shape)  # (150, 4)



# 2 모델
model = RandomForestClassifier(random_state=123)

# 3 훈련

model.fit(x_train, y_train)

# 4 평가, 예측
results = model.score(x_test, y_test)
print(x_train.shape)
print('model score : ', results)


# n_components = 4
# (120, 4)
# model score :  0.8

# n_components = 3
# (120, 3)
# model score :  0.8666666666666667

# n_components = 2
# (120, 2)
# model score :  0.8333333333333334