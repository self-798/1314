import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 加载数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 类别标签
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def show_sample_images(x_data, y_data, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_data[i])
        plt.title(class_names[y_data[i][0]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 显示训练集中的样本
print("训练集样本:")
show_sample_images(x_train, y_train)

# 显示一些基本信息
print(f"训练集形状: {x_train.shape}")
print(f"测试集形状: {x_test.shape}")
print(f"图片尺寸: {x_train[0].shape}")
print(f"像素值范围: [{x_train.min()}, {x_train.max()}]")