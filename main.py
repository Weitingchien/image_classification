import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np



from tensorflow.python.client import device_lib


def detect_overfitting(history):
    """
    檢測過擬合的函式。
    參數:
    history: 從 model.fit() 返回的 History
    """
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # 繪製訓練和驗證損失
    plt.figure(figsize=(12, 8))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 檢測過擬合
    # 找到驗證損失最小值的索引
    min_val_loss_idx = validation_loss.index(min(validation_loss))
    # 如果驗證損失最小值之後的損失有增加的趨勢，則認為可能過擬合
    if validation_loss[-1] > validation_loss[min_val_loss_idx]:
        print("可能發生了過擬合。")
    else:
        print("沒有明顯的過擬合跡象。")



def data_generator(X_data, y_data, batch_size, target_size):
    while True:
        for start in range(0, len(X_data), batch_size):
            end = min(start + batch_size, len(X_data))
            batch_images = []
            for i in range(start, end):
                image = tf.image.resize(X_data[i], target_size)
                image = image.numpy() / 255.0  # 將 Tensor 轉為 NumPy 陣列並進行正規化
                batch_images.append(image)
            yield np.array(batch_images), y_data[start:end]








def main():
    # 加載 CIFAR-100 數據集
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # 將標籤轉換為 one-hot 編碼
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    
    train_generator = data_generator(x_train, y_train, batch_size=64, target_size=(224, 224))
    test_generator = data_generator(x_test, y_test, batch_size=64, target_size=(224, 224))
    

    # 加載 ResNet50 模型，不包括頂層
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 凍結基礎模型的參數，這樣在訓練時它們不會被更新
    # base_model.trainable = False


    # 選擇要解凍的層
    # 這裡我們解凍最後的卷積塊（通常是最後的幾層）
    for layer in base_model.layers[-5:]:
        layer.trainable = True

    # 建立模型
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(100, activation='softmax')  # CIFAR-100 有 100 個類別
    ])

    # 設置模型如何學習的步驟
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # 印出模型的架構(印出模型每層的輸出形狀與參數的數量)
    model.summary()

    # 訓練模型的參數
    batch_size = 64
    epochs = 30



    # 訓練模型
    # history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    history = model.fit(
        train_generator,
        steps_per_epoch=x_train.shape[0] // 64,  # 每個 epoch 的步數
        epochs=30,
        validation_data=test_generator,
        validation_steps=x_test.shape[0] // 64  # 指定驗證步數
    )

    detect_overfitting(history)
    model_dir = 'trained'
    model.save(model_dir, 'ResNet50_CIFAR100.h5')

    # 評估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")


    # CIFAR-100 的類別名稱
    class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    show_predictions(model, x_test, y_test, class_names)

    


def show_predictions(model, x_test, y_test, class_names, num_images=5):
    # 從測試集中隨機選擇圖像
    indices = np.random.choice(range(len(x_test)), num_images)

    for i in indices:
        img = x_test[i]
        true_label = np.argmax(y_test[i])
        predicted_label = np.argmax(model.predict(img[np.newaxis, ...]))

        plt.figure()
        plt.imshow(img)
        plt.title(f"實際標籤: {class_names[true_label]}\n預測標籤: {class_names[predicted_label]}")
        plt.show()





if __name__ == '__main__':


    main()
