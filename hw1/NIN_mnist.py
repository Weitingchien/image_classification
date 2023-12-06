import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation


# MNISTSequence(Sequence)類別允許在訓練過程中動態載入和處理數據，減少記憶體的壓力, 因為一次resize訓練集所有圖, 記憶體會消耗太多
class MNISTSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, target_size=(224, 224)):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.target_size = target_size
    # 定義了這個序列的長度
    def __len__(self):
        # 計算整個數據集分為指定批次大小後的總批次數, np.ceil 確保即使最後一個批次不滿也會被計入
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    # 定義了如何獲取序列的一個批次
    def __getitem__(self, idx):
        # 從 self.x 中獲取第 idx 批次的特徵數據
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        # 從 self.y 中獲取第 idx 批次的標籤數據
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        """ 
        通過 np.expand_dims 在最後一個軸上增加一個維度
        ,將圖像從 28x28 變為 28x28x1, 以符合卷積層的輸入要求,
        然後將數據類型轉換為 float32, 並進行正歸化(除以255)
        """
        batch_x = np.expand_dims(batch_x, axis=-1).astype('float32') / 255.

        # 動態縮放圖像
        # 對 batch_x 中的每一張圖操作, 將結果轉換為 numpy array
        return np.array([tf.image.resize(image, self.target_size).numpy() for image in batch_x]), np.array(batch_y)




def create_nin_model():
    input_img = Input(shape=(224, 224, 1)) # 圖像的通道數, 1表示灰階

    """
    11x11 過濾器尺寸, 96個過濾器(卷稽核), 步長4: 過濾器在圖像上的移動步長為 4 像素, padding 1
    padding='same'：卷積層將在邊緣處填充，以保持輸出尺寸與輸入尺寸相同
    activate function: ReLU
    name='conv1'：為這個層命名為 'conv1'
    1x1 的卷積在這裡用於對特徵圖進行逐像素的學習
    MaxPooling2D: 最大池化層, 使用 3x3 的窗口大小和 2x2 的步長進行池化操作，以減少特徵圖的空間尺寸
    """

    # 第一個 NiN block
    conv1 = Conv2D(96, (11,11), strides=(4,4), padding='same', activation='relu', name='conv1')(input_img)
    conv1 = Conv2D(96, (1,1), padding='same', activation='relu', name='conv1_mlp1')(conv1)
    conv1 = Conv2D(96, (1,1), padding='same', activation='relu', name='conv1_mlp2')(conv1)
    pool1 = MaxPooling2D((3,3), strides=(2,2), padding='same', name='pool1')(conv1)

    # 第二個 NiN block
    # 5x5 過濾器尺寸, 256個過濾器(卷稽核), 步長1, padding 1
    conv2 = Conv2D(256, (5,5), strides=(1,1), padding='same', activation='relu', name='conv2')(pool1)
    conv2 = Conv2D(256, (1,1), padding='same', activation='relu', name='conv2_mlp1')(conv2)
    conv2 = Conv2D(256, (1,1), padding='same', activation='relu', name='conv2_mlp2')(conv2)
    pool2 = MaxPooling2D((3,3), strides=(2,2), padding='same', name='pool2')(conv2)

    # 第三個 NiN block
    # 3x3 卷積層尺寸, 384個過濾器(卷稽核), 步長1, padding 1
    conv3 = Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', name='conv3')(pool2)
    conv3 = Conv2D(384, (1,1), padding='same', activation='relu', name='conv3_mlp1')(conv3)
    conv3 = Conv2D(384, (1,1), padding='same', activation='relu', name='conv3_mlp2')(conv3)
    pool3 = MaxPooling2D((3,3), strides=(2,2), padding='same', name='pool3')(conv3)


    # 全域平均池化層: 對每個特徵圖進行平均池化，從而將每個特徵圖縮減為一個單一的數值
    global_pool = GlobalAveragePooling2D(name='global_avg_pool')(pool3)

    # Softmax 分類器: Softmax 用於多類別分類問題，可以將輸出轉換為機率分佈
    predictions = Activation('softmax', name='predictions')(global_pool)

    # 建立模型
    # 指定 input_img 為輸入層，predictions 為輸出層
    model = Model(inputs=input_img, outputs=predictions)

    return model



def main():
    model = create_nin_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()  # 印出模型架構
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    batch_size = 64
    target_size = (224, 224) # 決定要縮放的大小, 從28x28至224x224

    # 創建訓練和測試的序列生成器
    train_seq = MNISTSequence(x_train, y_train, batch_size, target_size)
    test_seq = MNISTSequence(x_test, y_test, batch_size, target_size)

    model = create_nin_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 使用序列生成器進行訓練
    model.fit(train_seq, validation_data=test_seq, epochs=10)


if __name__ == '__main__':
    main()
