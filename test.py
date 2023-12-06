from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
import numpy as np

# 加載 CIFAR-100 數據集
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 取出第一張圖像
img = x_train[0]

# 將圖像轉換為 uint8 型別
img = img.astype('uint8')

# 使用 PIL 對圖像進行上採樣
# 將圖像大小增加到 32 倍，使用 LANCZOS 插值
# 用來使模糊的圖像變得清晰
img_upsampled = Image.fromarray(img).resize((32*3, 32*3), Image.LANCZOS)

# 顯示圖像
plt.imshow(img_upsampled)
plt.show()