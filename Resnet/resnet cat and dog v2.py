import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 設定參數
TRAIN_DIR = "E:/vscode/deep learning/Resnet/training/"
TEST_DIR = "E:/vscode/deep learning/Resnet/test/"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5 #拉到只剩下五次是因為每次訓練時間太長，我真的懶得等待，晚餐吃完動畫看完結果還在跑訓練就很躁
NUM_CLASSES = 2  #貓與狗兩類

# 增強數據
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=30,  # 旋轉角度
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 載入訓練與測試數據
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 載入 ResNet-50 預訓練模型
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 凍結 ResNet 前面 80% 的層，讓高層參與訓練
for layer in model.layers[:int(len(model.layers) * 0.8)]:
    layer.trainable = False

# 建立新分類頭
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # 防止過擬合，每一次砍一半的神經元
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# 建立新模型
model = Model(inputs=model.input, outputs=predictions)

# 設定 AdamW 優化器 + 學習率衰減
optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)

# 編譯模型
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#設定回調函數
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),  #當val_loss 3 次沒改善時降低學習率
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  #提前停止，防止過擬合
]

#訓練模型
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks
)


model.save("第二版的分辨貓狗")
loss, accuracy = model.evaluate(test_generator)
print(f"測試損失: {loss:.4f}, 準確率: {accuracy:.2%}")


'''
測試損失: 0.6463, 準確率: 61.20%

阿不是哥們，我這沒有用的比較好欸
'''