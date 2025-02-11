import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
'''
ImageDataGenerator：資料增強 (Data Augmentation) 及預處理。
ResNet50：載入預訓練的 ResNet-50 模型。
Dense：全連接層 (Fully Connected Layer)。
Flatten：將多維輸出攤平成 1 維陣列。
GlobalAveragePooling2D：用於降低參數數量、加快訓練速度。
'''
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os


train_dir = r"E:\vscode\deep learning\Resnet\training" #訓練資料路徑
test_dir = r"E:\vscode\deep learning\Resnet\test" #測試資料路徑
'''
仔細想想 我是不是應該分成訓練兩個模型
一個分辨狗 一個分辨貓
'''

#圖片預處理（標準化 + 增強）
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 正規化到 [0,1] 範圍 把像素值 從 [0,255] 映射到 [0,1]，加快收斂
    rotation_range=30,  # 隨機旋轉30度 
    width_shift_range=0.2,  # 水平平移20%
    height_shift_range=0.2,  # 垂直平移20%
    shear_range=0.2,  # 隨機剪切變換 讓圖片斜向變形20%
    zoom_range=0.2,  # 隨機縮放隨機放大或縮小20%
    horizontal_flip=True,  # 水平翻轉 讓模型學會左右翻轉後仍能辨識
    fill_mode='nearest'  # 補充像素模式 當像素缺失時，填充最近的像素
)

test_datagen = ImageDataGenerator(rescale=1./255)  #測試集只需正規化

#加載圖片資料集
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # ResNet-50 需要 224x224 輸入 ，不過原始圖片不是224x224就是了
    batch_size=32,
    class_mode='binary'  # 兩個類別，貓跟狗，基本上是用於二元分類
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# 加載ResNet-50預訓練模型
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) #include_top=False：移除原本的全連接層，改為自定義分類層

#只訓練最後的分類層
for layer in base_model.layers:
    layer.trainable = False

    #保持 預訓練權重，不讓它們在訓練時被改變，減少訓練時間。

#添加自定義分類層
x = base_model.output
x = GlobalAveragePooling2D()(x)  # 全局平均池化 用於降維並保留重要特徵
x = Dense(512, activation='relu')(x)  # 全連接層，增加學習能力
x = Dense(1, activation='sigmoid')(x)  # sigmoid 適用於 二元分類 ，這是在做邏輯回歸的時候學的

# 建立完整模型
model = Model(inputs=base_model.input, outputs=x)

#編譯模型
model.compile(
    loss='binary_crossentropy',  # 適用於二元分類
    optimizer=Adam(learning_rate=0.0001),  # Adam 優化器，小學習率適合遷移學習
    metrics=['accuracy']
)

#訓練模型
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(test_generator)
)

#儲存模型
model.save("cats_vs_dogs_resnet50.h5")

#測試模型準確度
eval_result = model.evaluate(test_generator)
print(f"測試損失: {eval_result[0]:.4f}, 測試準確率: {eval_result[1]*100:.2f}%")
#初次結果如下，測試損失: 0.6403, 測試準確率: 63.75%

