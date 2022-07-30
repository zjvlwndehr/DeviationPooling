import numpy as np
import cv2
import tensorflow as tf
import os
import glob

class algo:
    def __init__(self):
        self.root = os.getenv('DATASET_PATH')
        self.mode = ''        
        self.path = self.root + '\\mnist\\*\\*.png'

        print(self.path)

        self.path = glob.glob(self.path)

        self.path = np.random.permutation(self.path)

        self.path = self.path[:100]
        print(self.path.shape)
        print('start loading data...')        
        self.x_indep = np.array([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in self.path])
        self.y_indep = np.array([int(i.split('\\')[-2]) for i in self.path])
        print('data loaded')

    def start(self):
        print('### original image ###')
        print(self.x_indep.shape)
        print(self.x_indep.dtype)
        print(self.x_indep.ndim)

        print('### deviation pooling ###')
        cnt = 0
        self.mode = 'd_pool'
        for data in self.x_indep:
            # data: (array), filter_size: (n, n), stride: (n, n)
            self.DeviationPooling2D(data, (2, 2), (1, 1), cnt)
            cnt += 1
        cnt = 0
        
        print('### max pooling ###')
        cnt = 0
        self.mode = 'm_pool'
        for data in self.x_indep:
            # data: (array), filter_size: (n, n), stride: (n, n)
            self.MaxPooling2D(data, (2, 2), (1, 1), cnt)
            cnt += 1
        cnt = 0

    def save_as_image(self, data, name, id):
        cv2.imwrite(f'./{self.mode}/{self.y_indep[id]}/{name}.png', data)
        print(f'./{self.mode}/{self.y_indep[id]}/{name}.png')
        pass

    # data: (array), filter_size: (n, n), stride: (n, n)
    def DeviationPooling2D(self, data, filter_size, stride, id):
        width, height = data.shape
        filter_width, filter_height = filter_size
        stride_width, stride_height = stride

        pool_width_size = int((width - filter_width) / stride_width) + 1
        pool_height_size = int((height - filter_height) / stride_height) + 1

        output = np.zeros(
            (pool_width_size, pool_height_size), dtype=data.dtype)
        print('output shape: ', output.shape)

        for stride_i in range(0, pool_height_size):
            i = stride_i * stride_height
            for stride_j in range(0, pool_width_size):
                j = stride_j * stride_width
                sample = data[i: i + filter_height, j: j + filter_width]
                deviation = []
                avr = np.mean(sample)
                for k in sample:
                    deviation.append(np.abs(k - avr))
                deviation = np.max(deviation).astype(data.dtype)
                output[stride_i, stride_j] = deviation

        print(f'output shape: {output.shape}, output.dtype: {output.dtype}, output.ndim: {output.ndim}')
        self.save_as_image(output, f'deviation_pooling_{id}', id)

    # data: (array), filter_size: (n, n), stride: (n, n)
    def MaxPooling2D(self, data, filter_size, stride, id):
        width, height = data.shape
        filter_width, filter_height = filter_size
        stride_width, stride_height = stride

        pool_width_size = int((width - filter_width) / stride_width) + 1
        pool_height_size = int((height - filter_height) / stride_height) + 1

        output = np.zeros(
            (pool_width_size, pool_height_size), dtype=data.dtype)
        print('output shape: ', output.shape)

        for stride_i in range(0, pool_height_size):
            i = stride_i * stride_height
            for stride_j in range(0, pool_width_size):
                j = stride_j * stride_width
                sample = data[i: i + filter_height, j: j + filter_width]
                max = np.max(sample).astype(data.dtype)
                output[stride_i, stride_j] = max

        print(f'output shape: {output.shape}, output.dtype: {output.dtype}, output.ndim: {output.ndim}')
        self.save_as_image(output, f'max_pooling_{id}', id)


if __name__ == "__main__":
    obj = algo()
    obj.start()
