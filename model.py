import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import cv2

class MODEL:
    def __init__(self):
        self.epochs, self.batrch_size, self.imageSize = 10, 8, (28,28)
        self.input_shape = self.imageSize + (1,)
        # 풀링된 이미지를 미리 불러옴 './m_pool'(max pooling) 또는 './d_pool'(deviation pooling)
        root_path = './m_pool'
        
        print(self.input_shape)

        # shuffle the paths
        self.paths = np.random.permutation(glob.glob(root_path + "/*/*.png"))
        print(self.paths[:10])
        
        print("start loading")
        # (27,27) to (28,28)
        # 3 channel to 1 channel
        # load image
        self.x_indep = np.array([cv2.cvtColor(cv2.resize(cv2.imread(path), self.imageSize, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY) for path in self.paths])
        print("loaded")
        print(self.x_indep.shape)

        # labeling
        self.y_depen = np.array([path.split("\\")[-2] for path in self.paths])
        self.k_classes = len(set(self.y_depen))
        print(self.k_classes)
    

    # 훈련전 우리는 이미지를 미리 풀링하였음.
    # 모델의 summary를 통해서 풀링 과정을 확인할 수는 없으나
    # algorithm.py에서 훈련전 이미지를 미리 풀링하는 과정을 확인할 수 있음.
    # 따라서 풀링 과정이 추가된 모델로 보아야함.
    # create model using maxpooling
    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=self.k_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # train model
    def train_model(self):
        # reshape image
        self.x_indep = self.x_indep.reshape(self.x_indep.shape[0], 28, 28, 1)
        # one-hot encoding
        self.y_depen = pd.get_dummies(self.y_depen)
        print(self.x_indep.shape, self.y_depen.shape)
        print(self.x_indep.ndim)

        model = self.create_model()

        model.fit(self.x_indep, self.y_depen, epochs=self.epochs, batch_size=self.batrch_size)
        # m_model.h5(max pooling을 사용한 모델) 또는 d_model.h5(deviation pooling을 사용한 모델)으로 저장
        model.save("./model/m_model.h5")
        print("model saved")

if __name__ == "__main__":
    obj = MODEL()
    obj.train_model()
