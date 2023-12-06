import keras
import numpy as np
import matplotlib.pyplot as plt 
np.random.seed(10)

# 2.下載 mnist data
from keras.datasets import mnist




# 2.建立 plot_image 函數顯示數字影像 

def plot_images_labels_predict(images, labels, prediction, idx, num=10):  
    fig = plt.gcf()  
    fig.set_size_inches(12, 14)  
    if num > 25: num = 25  
    for i in range(0, num):  
        ax=plt.subplot(5,5, 1+i)  
        ax.imshow(images[idx], cmap='binary')  
        title = "l=" + str(labels[idx])  
        if len(prediction) > 0:  
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))  
        else:  
            title = "l={}".format(str(labels[idx]))  
        ax.set_title(title, fontsize=10)  
        ax.set_xticks([]); ax.set_yticks([])  
        idx+=1  
    plt.show()  

def main():
    # 3.讀取與查看 mnist data
    (X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()  
    print("\t[Info] train data={:7,}".format(len(X_train_image)))  
    print("\t[Info] test  data={:7,}".format(len(X_test_image)))

    # 1.訓練資料是由 images 與 labels 所組成 
    print("\t[Info] Shape of train data=%s" % (str(X_train_image.shape)))  
    print("\t[Info] Shape of train label=%s" % (str(y_train_label.shape)))


    # 3.執行 plot_image 函數查看第 0 筆數字影像與 label 資料 
    plot_images_labels_predict(X_train_image, y_train_label, [], 0, 10)

    #Normalizing the data
    X_train = X_train_image.astype('float32') / 255.0
    X_test = X_test_image.astype('float32') / 255.0

    #Initializing model
    model = keras.models.Sequential()

    #Adding the model layers
    model.add(keras.layers.LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(128))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation='softmax'))

    #Compiling the model
    model.compile( loss='sparse_categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001), metrics=['accuracy'] )

    #Fitting data to the model
    history = model.fit(X_train, y_train_label, epochs=3, validation_data=(X_test, y_test_label))

if __name__ == "__main__":
    main()
