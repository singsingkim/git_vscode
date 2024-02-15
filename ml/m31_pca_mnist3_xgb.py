# xgboost와 그리드서치, 랜덤서치, Halving 등을 사용

# n_jobs = -1

# tree_method = 'gpu_hist',
# predictor = 'gpu_predictor',
# gpu_id=0,

# m31_2 번보다 성능을 좋게 만든다

import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import time
import warnings

# 경고 메시지를 무시하도록 설정
warnings.filterwarnings("ignore", category=UserWarning)

# MNIST 데이터셋 로드
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X = X / 255.0  # 데이터 스케일링

# 훈련, 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA 차원 범위 설정
pca_dims = [154, 331, 486, 713, 784]

for n_components in pca_dims:
    print(f"PCA 차원: {n_components}")

    # PCA 적용
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # XGBoost 데이터셋 생성
    dtrain = xgb.DMatrix(X_train_pca, label=y_train)
    dtest = xgb.DMatrix(X_test_pca, label=y_test)

    # XGBoost 모델 설정 - GPU 사용
    param = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'multi:softmax',
        'num_class': 10,
        'tree_method': 'gpu_hist',  # GPU를 사용하기 위한 설정
        'verbose': 0  # 출력 최소화 설정

    }

    # 모델 훈련
    num_round = 50
    start_time = time.time()
    bst = xgb.train(param, dtrain, num_round)
    training_time = time.time() - start_time

    # 예측 수행
    preds = bst.predict(dtest)

    # 정확도 계산
    accuracy = accuracy_score(y_test, preds)
    print(f"테스트 정확도: {accuracy}")
    print(f"훈련 시간: {training_time:.2f}초")
    print()

# PCA 차원: 154
# 테스트 정확도: 0.0
# 훈련 시간: 2.70초

# PCA 차원: 331
# 테스트 정확도: 0.0
# 훈련 시간: 4.29초

# PCA 차원: 486
# 테스트 정확도: 0.0
# 훈련 시간: 6.07초

# PCA 차원: 713
# 테스트 정확도: 0.0
# 훈련 시간: 8.15초

# PCA 차원: 784
# 테스트 정확도: 0.0
# 훈련 시간: 8.83초