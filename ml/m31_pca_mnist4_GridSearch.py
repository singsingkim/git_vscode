from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import warnings
import xgboost as xgb

# 경고 메시지를 무시하도록 설정
warnings.filterwarnings("ignore", category=UserWarning)

# MNIST 데이터셋 로드
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X = X / 255.0  # 데이터 스케일링

# 훈련, 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 문자열 형태의 레이블을 정수로 변환
y_train = y_train.astype(int)
y_test = y_test.astype(int)


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
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.1, 0.3, 0.5],
        'n_estimators': [50, 100, 150],
        'tree_method': ['gpu_hist'],  # GPU를 사용하기 위한 설정
        'num_class': [10],
        'verbosity' : [0]
    }

    # 그리드 서치를 위한 모델 정의
    xgb_model = xgb.XGBClassifier()

    # 그리드 서치 수행
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train_pca, y_train)

    # 최적의 모델 및 파라미터 출력
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("최적의 파라미터:", best_params)

    # 예측 수행
    preds = best_model.predict(X_test_pca)

    # 정확도 계산
    accuracy = accuracy_score(y_test, preds)
    print(f"테스트 정확도: {accuracy}")
    print()
