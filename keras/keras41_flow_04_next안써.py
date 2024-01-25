from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest',
)

augument_size = 100

print(x_train[0].shape) # (28, 28)
# plt.imshow(x_train[0])
# plt.show()

x_data = train_datagen.flow(    # x 와 y 가 들어간다
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1),
                                # 전체가 x  / np.tile 은 합쳐주는 역할. 처음(100, 784) 에서 최종적으로 (100, 28, 28, 1) 
    np.zeros(augument_size),     # 전체가 y / 구색만 갖추기위해 위 처럼 설정 없이 전부 0 
    batch_size=augument_size,
    shuffle=False

)#.next()    # 아규먼트 사이즈가 10이라면 (10, 28,28,1) 짜리가 x_data 에 첫번째에 들어간다


print(x_data)
#### print(x_data.shape) # 튜플형태라서 에러임
                    # 왜냐하면 flow에서 튜플형태로 반환했다
                    # 리스트나 딕셔너리는 쉐이프로 확인을 못한다
print(x_data[0][0].shape)  # (100,28,28,1)
print(x_data[0][1].shape)  # (100,)



print(np.unique(x_data[0][1],return_counts=True))
print(x_data[0][0][1].shape)    # (28, 28, 1)


# 그림그리기
plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i],cmap='gray') # 위에서 next() 를 넣어주면 첫칸 배치 순서를 정하지 않아도 된다. 
plt.show()

