from pickletools import uint8
from random import sample
import numpy as np
import cv2
import tensorflow as tf

class algo:
    def __init__(self):
        self.root = './'
        self.path = self.root + 'image/' + 'sample.png'
        self.data = cv2.imread(self.path)
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
    
    def main(self):
        print('### original image ###')
        print(self.data.shape)
        print(self.data.dtype)
        print(self.data.ndim)
        print(self.data)

        print('### deviation pooling ###')
        self.DeviationPooling2D(self.data, (2, 2), (2, 2))

    def save_as_image(self, data, name):
        cv2.imwrite(self.root + 'output/' + name + '.png', data)
        print('save as ' + name + '.bmp')
    
    def DeviationPooling2D(self, data, filter_size, stride): # data: (array), filter_size: (n, n), stride: (n, n)
        width, height = data.shape
        filter_width, filter_height = filter_size
        stride_width, stride_height = stride

        pool_width_size = int((width - filter_width) / stride_width) + 1
        pool_height_size = int((height - filter_height) / stride_height) + 1

        # output = np.empty((pool_width_size, pool_height_size), dtype=data.dtype)
        output = np.zeros((pool_width_size, pool_height_size), dtype=data.dtype)
        K = filter_width * filter_height
        print('output shape: ', output.shape)

        for stride_i in range(0,pool_height_size):
            i = stride_i * stride_height
            cnt = 0
            for stride_j in range(0,pool_width_size):
                j = stride_j * stride_width 
                sample = data[i : i + filter_height, j : j + filter_width]
                deviation = []
                avr = np.mean(sample)
                for k in sample:
                    deviation.append(np.abs(k - avr))
                deviation = np.max(deviation).astype(data.dtype)
                output[stride_i, stride_j] = deviation
            
        print(output.shape)
        print(output.dtype)
        print(output.ndim)
        print(output)
        self.save_as_image(output, 'deviation_pooling')
    
    def MaxPooling2D(self, data, filter_size, stride):
        X = tf.keras.layers.Input(shape=(data.shape))
        pool = tf.keras.layers.MaxPooling2D(pool_size=filter_size, strides=stride)(X)


        
        
if __name__ == "__main__":
    obj = algo()
    obj.main()