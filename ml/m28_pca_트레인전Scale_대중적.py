# 주성분분석이라 부름
# 스케일링 후 pca 후 트레인
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


scaler = StandardScaler()
x = scaler.fit_transform(x)


pca = PCA(n_components=4)
x = pca.fit_transform(x)
print(x)
print(x.shape)  # (150, 4)

x_train, x_test, y_train, y_test = train_test_split(
    
    x, y, train_size=0.8, random_state=123, shuffle=True, stratify=y
)

# 2 모델
model = RandomForestClassifier(random_state=123)

# 3 훈련

model.fit(x_train, y_train)

# 4 평가, 예측
results = model.score(x_test, y_test)
print(x.shape)
print('model score : ', results)