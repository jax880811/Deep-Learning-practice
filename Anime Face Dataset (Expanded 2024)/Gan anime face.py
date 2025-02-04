import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# 資料參數
IMAGE_DIR = r"E:\vscode\deep learning\Anime Face Dataset (Expanded 2024)\faces"
IMAGE_SHAPE = (96, 96, 3)  # 圖片大小
LATENT_DIM = 100  # 噪聲向量的維度
MODEL_NAME = "GAN_anime_face_model"  # 模型名稱
OUTPUT_DIR = r"E:\vscode\deep learning\Anime Face Dataset (Expanded 2024)\圖片觀察"  # 圖片觀察儲存位置

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定義生成器
def build_generator():
    model = Sequential()
    model.add(Dense(12 * 12 * 256, activation="relu", input_dim=LATENT_DIM))  # 初始大小調整為 12x12
    model.add(Reshape((12, 12, 256)))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"))  # 12x12 -> 24x24
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"))  # 24x24 -> 48x48
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation="tanh"))  # 48x48 -> 96x96
    return model
'''
生成器的作用：從隨機噪聲向量生成假圖像。
Domain Knowledge：生成器學習數據分佈，產生與真實圖像相似的圖像。
Dense：將輸入的隨機向量映射到高維特徵空間（12x12x256）。
Reshape：將平坦向量重塑為 12x12x256 的張量。
BatchNormalization：標準化輸出，穩定生成器的訓練。
Conv2DTranspose：轉置卷積層用於逐步放大圖像尺寸（12x12 -> 24x24 -> 48x48 -> 96x96）。
激活函數：
relu：保持非線性，適合生成階段。
tanh：輸出範圍為 [-1, 1]，與圖像的正規化範圍對應。
'''

# 定義判別器
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=IMAGE_SHAPE))  # 96x96 -> 48x48
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))  # 48x48 -> 24x24
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))  # 24x24 -> 12x12
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return model
'''
判別器的作用：區分輸入圖像是真實還是生成的。
Domain Knowledge：作為一個二分類模型，判別器為生成器提供方向性反饋。
Conv2D：提取圖像特徵，逐步減小尺寸以增加特徵表示的抽象程度。
LeakyReLU：允許負斜率，解決 ReLU 的梯度消失問題。
Dropout：隨機丟棄神經元，防止過擬合。
Flatten：將多維特徵展平成一維，用於輸出層。
Dense：輸出一個標量（範圍 [0, 1]），表示真實性。
'''

# 定義 GAN 模型
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(LATENT_DIM,))
    fake_image = generator(gan_input)
    gan_output = discriminator(fake_image)
    return Model(gan_input, gan_output)

'''
GAN的結構：生成器 + 判別器。
生成器生成假圖像，判別器評估其真實性。
discriminator.trainable = False：凍結判別器權重，避免在GAN訓練過程中被更新。
Input：定義GAN的輸入為噪聲向量。
gan_output：生成器的輸出直接傳給判別器。
'''
# 加載資料
def load_images(image_dir, image_shape):
    images = []
    for file_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, file_name)
        img = load_img(img_path, target_size=image_shape[:2])  # 圖片大小保持為 96x96
        img_array = img_to_array(img) / 127.5 - 1.0  # 正規化到 [-1, 1]
        images.append(img_array)
    return np.array(images)

# 保存生成圖片
def save_generated_images(generator, epoch, output_dir):
    noise = np.random.normal(0, 1, (36, LATENT_DIM))  # 隨機生成 36 張圖片
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) * 127.5  # 轉換到 [0, 255]
    generated_images = generated_images.astype(np.uint8)

    # 繪製 6x6 圖片方格
    fig, axs = plt.subplots(6, 6, figsize=(12, 12))
    cnt = 0
    for i in range(6):
        for j in range(6):
            axs[i, j].imshow(generated_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"epoch_{epoch}.png")
    plt.savefig(output_path)
    plt.close()

# 訓練模型
def train_gan(epochs, batch_size, save_path):
    # 建立模型
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"])
    gan = build_gan(generator, discriminator)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy")

    # 加載資料
    dataset = load_images(IMAGE_DIR, IMAGE_SHAPE)
    half_batch = batch_size // 2

    for epoch in range(1, epochs + 1):
        # 訓練判別器
        idx = np.random.randint(0, dataset.shape[0], half_batch)
        real_images = dataset[idx]
        noise = np.random.normal(0, 1, (half_batch, LATENT_DIM))
        fake_images = generator.predict(noise)

        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 訓練生成器
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)


        # 每隔 100 個 epoch 打印損失
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

        # 每隔 100 個 epoch 保存生成圖片
        if epoch % 100 == 0:
            save_generated_images(generator, epoch, OUTPUT_DIR)
            generator.save(save_path)

'''
noise = np.random.normal(...)：生成新的噪聲向量。
valid_labels = np.ones(...)：
訓練生成器時，假裝生成的圖像是真實的，標籤為 1。
為什麼設置為真實：生成器的目標是欺騙判別器，因此需要使用這樣的標籤。
g_loss = gan.train_on_batch(...)：使用 GAN 模型訓練生成器。
Domain Knowledge：生成器的梯度是從判別器的輸出誤差反向傳遞來的，這是一個對抗過程。
'''

'''
idx = np.random.randint(...)：從資料集中隨機選取 half_batch 張真實圖像。
為什麼隨機選擇：避免過擬合，讓判別器學習數據的多樣性。
real_images = dataset[idx]：根據索引提取真實圖像。
noise = np.random.normal(...)：生成服從標準正態分佈的隨機噪聲向量。
Domain Knowledge：噪聲向量作為生成器輸入，用於生成多樣性圖像。
fake_images = generator.predict(noise)：生成器將噪聲轉換為假圖像。
real_labels = np.ones(...) / fake_labels = np.zeros(...)：
真實圖像的標籤為 1，假圖像的標籤為 0。
為什麼要標記：判別器需要這些標籤進行監督學習。
d_loss_real = discriminator.train_on_batch(real_images, real_labels)：
判別器用真實圖像進行一次訓練。
d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)：
判別器用假圖像進行一次訓練。
d_loss = 0.5 * np.add(...)：計算真實和假樣本的平均損失。
為什麼計算平均值：平衡判別器對真實與假樣本的學習效果。
'''

'''
generator = build_generator()：建立生成器模型，負責生成假圖像。
Domain Knowledge：生成器的目的是學習數據分佈，並生成逼真的樣本以騙過判別器。
discriminator = build_discriminator()：建立判別器模型，負責區分真實和生成圖像。
Domain Knowledge：判別器的作用是提供生成器的訓練信號，幫助其提高生成樣本的真實性。
discriminator.compile(...)：使用 Adam 優化器編譯判別器，損失函數為二元交叉熵（binary_crossentropy）。
為什麼用 Adam：適合處理稀疏梯度，能穩定GAN的訓練。
為什麼用 binary_crossentropy：判別器是二分類模型（真實 vs 假），此損失函數與任務需求吻合。
gan = build_gan(generator, discriminator)：將生成器和判別器結合，構建 GAN 模型。
gan.compile(...)：使用相同的優化器和損失函數編譯 GAN 模型。
Domain Knowledge：GAN 的目標是最小化生成器生成的樣本被判別器判為假的概率（對抗訓練）。
'''
# 執行訓練
train_gan(epochs=5000, batch_size=64, save_path=f"{MODEL_NAME}.h5")