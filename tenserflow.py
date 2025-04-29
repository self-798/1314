import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 创建日志目录
log_dir = './logs/cifar10'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 回调函数
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10),
    callbacks.ModelCheckpoint('best_cifar10_model.keras', save_best_only=True),
    callbacks.ReduceLROnPlateau(factor=0.2, patience=5),
    callbacks.TensorBoard(log_dir=log_dir)
]
# 加载CIFAR-10数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.1,
    validation_split=0.2
)

# 构建深度CNN模型
model = models.Sequential([
    # 第一组卷积层
    layers.Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # 第二组卷积层
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # 第三组卷积层
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # 全连接层
    layers.Flatten(),
    layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 学习率调度
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.95
)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 回调函数
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10),
    callbacks.ModelCheckpoint('best_cifar10_model.keras', save_best_only=True),
    callbacks.ReduceLROnPlateau(factor=0.2, patience=5),
    callbacks.TensorBoard(log_dir='./logs/cifar10')
]

# 训练模型
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32, subset='training'),
    validation_data=datagen.flow(x_train, y_train, batch_size=32, subset='validation'),
    epochs=50,
    callbacks=callbacks_list,
    verbose=1
)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nCIFAR-10测试集准确率: {test_acc:.4f}')