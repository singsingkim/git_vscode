'''
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

# m31_1 에서 뽑은 4가지 결과로
# 4가지 모델을 맹그러
# input_shape = ()
# 1. 70000, 154
# 2. 70000, 331
# 3. 70000, 486
# 4. 70000, 713
# 5. 70000, 784 원본

# 시간과 성능을 체크한다
# 결과 예시

# 결과 1.  PCA=154
# 걸린시간 0000초
# acc = 0.0000

# 결과 2.  PCA=331
# 걸린시간 0000초
# acc = 0.0000

# 결과 3.  PCA=486
# 걸린시간 0000초
# acc = 0.0000

# 결과 4.  PCA=713
# 걸린시간 0000초
# acc = 0.0000

# 결과 5.  PCA=784 원본
# 걸린시간 0000초
# acc = 0.0000

'''

"""
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 합치기
x = np.concatenate([x_train, x_test], axis=0)   # (70000, 28, 28)

x = x.reshape(x.shape[0], -1)
print(x.shape)  # (70000, 784)

# 데이터 표준화
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca_dims = [154, 331, 486, 713, 784]

for n_components in pca_dims:
    print(f"PCA={n_components}")
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x_scaled)

    # 데이터 분할
    x_train, x_test, _, _ = train_test_split(x_pca, np.zeros(x_pca.shape[0]), test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 생성
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 모델 학습 시간 측정 시작  
    start_time = time.time()

    # 모델 학습
    clf.fit(x_train, y_train)

    # 학습 시간 계산
    training_time = time.time() - start_time

    # 정확도 평가
    accuracy = clf.score(x_test, y_test)

    print(f"걸린시간 {training_time:.4f}초")
    print(f"acc = {accuracy:.4f}")
    print()
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리 및 준비
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# 원-핫 인코딩
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

# PCA 차원 수 범위
pca_dims = [154, 331, 486, 713, 784]

for n_components in pca_dims:
    print(f"PCA 차원: {n_components}")

    # PCA 적용
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # DNN 모델 생성
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(n_components,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 훈련 시간 측정 시작
    start_time = time.time()

    # 모델 훈련
    model.fit(x_train_pca, y_train, epochs=10, batch_size=128, validation_data=(x_test_pca, y_test), verbose=0)
                    # 따라서, validation_data는 모델 학습 중에 모델의 성능을 평가하기 위한 데이터를 지정하는 데 사용되고, 
                    # split은 데이터를 훈련과 테스트로 나누는 데 사용됩니다.


    # 훈련 시간 계산
    training_time = time.time() - start_time

    # 모델 평가
    loss, accuracy = model.evaluate(x_test_pca, y_test, verbose=0)
    print(f"테스트 손실: {loss:.4f}")
    print(f"테스트 정확도: {accuracy:.4f}")
    print(f"걸린 시간: {training_time:.2f}초")
    print()


# PCA 차원: 154
# 테스트 손실: 0.1152
# 테스트 정확도: 0.9758
# 걸린 시간: 6.37초

# PCA 차원: 331
# 테스트 손실: 0.1211
# 테스트 정확도: 0.9764
# 걸린 시간: 5.55초

# PCA 차원: 486
# 테스트 손실: 0.1244
# 테스트 정확도: 0.9751
# 걸린 시간: 5.74초

# PCA 차원: 713
# 테스트 손실: 0.1317
# 테스트 정확도: 0.9742
# 걸린 시간: 5.79초

# PCA 차원: 784
# 테스트 손실: 0.1206
# 테스트 정확도: 0.9757
# 걸린 시간: 5.78초

# PS C:\Study> c



